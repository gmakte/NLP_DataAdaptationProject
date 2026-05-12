########### IMPORTS #####################
# NOTE: run this script from the root directory of the project to avoid path issues 

import argparse
import os
import json
import re
from util.preprocessing import parse_iob2_file
from eval import evaluate_predictions

import time
from datetime import timedelta

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI

#################### ---------HELPER FUNCTIONS--------- ################################
# TODO: fix annotate_chunk into smaller pieces:
# generate_raw_output()
# parse_output()
# validate_output()

# different model configurations
def load_model(model_name, quantized=False):

    if model_name == "gpt": #configuration is slightly different for API based models
        client = OpenAI()
        return {
            "backend": "openai",
            "client": client,
            "model_name": "gpt-5.4"
        }

    # if not gpt, continue loading a local model using Hugging Face Transformers
    elif model_name == "mistral":
        hf_name = "mistralai/Mistral-7B-Instruct-v0.3"

    elif model_name == "llama":
        hf_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    else:
        raise ValueError(f"Unsupported model name: {model_name}")


    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model_kwargs = {
    "device_map": "auto",
    "trust_remote_code": True,
    "use_safetensors": True
}

    if quantized:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        print("Loading model in 4-bit quantized mode...\n")

        model_kwargs["quantization_config"] = bnb_config

    else:
        print("Loading model in full precision mode...\n")
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(hf_name, **model_kwargs)
    model.eval()

    return {
        "backend": "hf",
        "model_name": hf_name,
        "model": model,
        "tokenizer": tokenizer
    }


# create flattened lists of sentences of max length to parse in one iteration through the model,
# and keep a track of the original sentence boundaries to be able to reconstruct the output into sentences after annotation
def build_chunks(sentences, labels=None, max_chunk_size=250): #usable both in validation or annotation mode according to labels
    chunks = []
    current_chunk = {
        "size": 0,
        "tokens": [], 
        "sent_lengths": [], 
        "sent_indices": [], 
        "gold_labels": [] if labels else None}

    for i, sent in enumerate(sentences): # remember sent is a list of tokens
        sent_len = len(sent) # how are we gonna handle extremely long sentences? for now they are treated separately as their own chunk

        # if adding the next sentence would exceed the max chunk size, finalize the current chunk and start a new one
        if current_chunk["size"] + sent_len > max_chunk_size:
            chunks.append(current_chunk)

            current_chunk = {
                "size": 0,
                "tokens": [], 
                "sent_lengths": [], 
                "sent_indices": [], 
                "gold_labels": [] if labels else None}

        # add sentence metadata to  current chunk
        current_chunk["tokens"].extend(sent)
        current_chunk["size"] += sent_len
        current_chunk["sent_lengths"].append(sent_len)
        current_chunk["sent_indices"].append(i)

        if labels:
            current_chunk["gold_labels"].append(labels[i])

    # add any remaining sentences as the last chunk
    if current_chunk["tokens"]:
        chunks.append(current_chunk)

    return chunks


# style prompt, ready to inject specific input text
def format_prompt(template_path, example_path):
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    with open(example_path, "r", encoding="utf-8") as f:
        example_input_lines = []
        example_output_lines = []

        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            _, token, label = line.split('\t')
            example_input_lines.append(token)
            example_output_lines.append(f"{token}\t{label}")

    example_input = '\n'.join(example_input_lines)
    example_output = '\n'.join(example_output_lines)

    # inject into prompt template
    prompt_template = template.format(
        example_input=example_input,
        example_output=example_output
    )

    return prompt_template

# align predictions and handle mismatches to avoid data shifting issues
def align_pred_pairs_to_chunk(pred_pairs, chunk):
    """
    Aligns model output [[pred_token, pred_label], ...]
    to chunk["tokens"].

    If a gold token is missing from predictions, assign MISMATCH.
    Returns flat labels aligned to chunk["tokens"].
    """

    gold_tokens = chunk["tokens"]
    aligned_labels = []

    pred_i = 0

    for gold_i, gold_token in enumerate(gold_tokens):

        # no more predictions left
        if pred_i >= len(pred_pairs):
            aligned_labels.append("MISMATCH")
            continue

        pred_token, pred_label = pred_pairs[pred_i]

        # perfect match
        if pred_token == gold_token:
            aligned_labels.append(pred_label)
            pred_i += 1

        # if current gold token was skipped by model
        # and predicted token matches next gold

        elif (gold_i + 1 < len(gold_tokens) and pred_token == gold_tokens[gold_i + 1]):
            aligned_labels.append("MISMATCH")
            # do NOT advance pred_i
            # next loop will match this pred_token to next gold token

        # otherwise token mismatch
        else:
            aligned_labels.append("MISMATCH")
            pred_i += 1

    mismatch_count = aligned_labels.count("MISMATCH")

    print(
        f"Recovered chunk with "
        f"{mismatch_count} mismatches "
        f"out of {len(aligned_labels)} tokens."
    )

    return aligned_labels
        

def parse_tsv_output(outputs):
    if not outputs.strip():
        raise ValueError("Empty model output")
    
    pred_pairs = []
    lines = outputs.splitlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split("\t")

        # perfectly formatted
        if len(parts) == 2:
            token, label = parts

        # missing label
        elif len(parts) == 1:
            token = parts[0]
            label = "MISMATCH"
        
        # too many columns
        else:
            token = parts[0]
            label = "MISMATCH"

        pred_pairs.append([token, label])

    return pred_pairs


# use the model to annotate a chunk of tokens and return the predicted labels as a 
def annotate_chunk(prompt_template, chunk, model_info, temperature):
    chunk_tokens = '\n'.join(chunk["tokens"])
    prompt = prompt_template.format(test_input=chunk_tokens)

    pred_pairs = None

    for attempt in range(3):

        # model 1 (gpt 5.4)
        if model_info["backend"] == "openai":
            # prepare the prompt for the open ai API 

            print(f"Sending prompt to {model_info['model_name']}, attempt {attempt+1}...\n")
            response = model_info["client"].chat.completions.create(
                model=model_info["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )

            # extract the predicted labels from the response
            outputs = response.choices[0].message.content.strip()


        # model 2 (llama, mistral)
        elif model_info["backend"] == "hf":
            # prepare the input for the Hugging Face model
            tokenizer = model_info["tokenizer"]
            inputs = tokenizer(prompt, return_tensors="pt").to(model_info["model"].device)

            print(f"Sending prompt to {model_info['model_name']}, attempt {attempt+1}...\n")
            with torch.no_grad():
                outputs = model_info["model"].generate(**inputs, max_new_tokens=chunk["size"]*8, do_sample=False)

            input_len = inputs["input_ids"].shape[-1]
            generated_tokens = outputs[0][input_len:]
            outputs = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # model 3, invalid
        else:
            raise ValueError(f"Unsupported model name: {model_info['model_name']}. Supported options are: mistral, llama, gpt.")
        
                
        # parse output to valid list, otherwise retry with a more explicit prompt
        try:
            pred_pairs = parse_tsv_output(outputs)
            break

        except ValueError as e:

            print(f"\nParsing failed on attempt {attempt+1}")
            print(f"Error: {e}")
            print(f"Raw output:\n{outputs}")

            if attempt == 2:
                raise ValueError(
                    f"Maximum attempts reached.\n"
                    f"Failed to parse model output.\n\n"
                    f"Raw output:\n{outputs}"
                )

            with open("annotation/prompt_retry.txt", "r", encoding="utf-8") as f:
                retry_message = f.read()
            
            prompt += retry_message.format(e=str(e))
            time.sleep(2)

    aligned_labels = align_pred_pairs_to_chunk(pred_pairs, chunk)
    return aligned_labels


def reconstruct_nested(flat_labels, chunk):
    nested_tokens = []
    nested_labels = []

    token_start = 0

    for sent_len in chunk["sent_lengths"]:
        token_end = token_start + sent_len

        sentence_tokens = chunk["tokens"][token_start:token_end]
        sentence_labels = flat_labels[token_start:token_end]

        nested_tokens.append(sentence_tokens)
        nested_labels.append(sentence_labels)

        token_start = token_end

    return nested_tokens, nested_labels


def write_iob2(sentences, labels, output_path):
    with open(output_path, "a", encoding="utf-8") as f:
        for sent_tokens, sent_labels in zip(sentences, labels):

            for i, (token, label) in enumerate(
                zip(sent_tokens, sent_labels),
                start=1
            ):

                f.write(f"{i}\t{token}\t{label}\n")

            # blank line between sentences
            f.write("\n")


##### ----------------------------- MAIN SCRIPT -------------------------------------- #####

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="mistral")
    parser.add_argument("--quantized", action="store_true", help="Whether to load the model in 4-bit quantized mode for memory efficiency.")
    parser.add_argument("--mode", default="validate")
    parser.add_argument("--create_chunks", action="store_true")
    parser.add_argument("--max_chunk_size", type=int, default=250, help="Maximum number of words to include in each chunk sent to the model for annotation.")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--input_data", default="annotation/FIN5_validation.txt")
    parser.add_argument("--output_dir", default="annotation/validation")
    parser.add_argument("--output_file", required=True)

    args = parser.parse_args()
    model_name = args.model_name.lower()
    quantized = args.quantized

    model = load_model(model_name, quantized)

    # load validation data
    if args.create_chunks:

        sentences, true_labels = parse_iob2_file(args.input_data)

        print("Creating chunks from sentences...\n")
        chunks = build_chunks(
            sentences,
            labels=true_labels,
            max_chunk_size=args.max_chunk_size
        )

        # since the output of creating chunks is deterministic, we can save it to reuse 
        with open("annotation/chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)

    else:
        print("Loading chunks from file...\n")
        with open("annotation/chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)

    
    print(f"Loading prompt template...\n")
    prompt_template = format_prompt("annotation/prompt_annotation.txt", "annotation/one_shot.txt")

    # create output dir from args
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    
    # clear previous file
    open(output_path, "w").close()

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        chunk_labels = annotate_chunk(prompt_template, chunk, model, args.temperature) # these are flat labels for the chunk
        nested_tokens, nested_labels = reconstruct_nested(chunk_labels, chunk)

        # save to file progressively
        write_iob2(nested_tokens, nested_labels, output_path) 

    print(f"Results saved to {output_path}")