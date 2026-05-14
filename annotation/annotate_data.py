########### IMPORTS #####################
# NOTE: run this script from the root directory of the project to avoid path issues 

import argparse
import os
import json
import re

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import spacy

from util.preprocessing import parse_iob2_file

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
def build_chunks(sentences, labels=None, max_chunk_size=50): #usable both in validation or annotation mode according to labels
    chunks = []
    current_chunk = {
        "size": 0,

        # flattened structure of tokens and labels
        "tokens": [], 
        "flat_gold_labels": [] if labels else None,

        # nested structure of sentences and labels
        "sentences": [],
        "gold_labels": [] if labels else None,

        # metadata used for reconstruction
        "sent_lengths": [], 
        "sent_indices": []}

    for i, sent in enumerate(sentences): # remember sent is a list of tokens
        sent_len = len(sent) # how are we gonna handle extremely long sentences? for now they are treated separately as their own chunk

        # if adding the next sentence would exceed the max chunk size, finalize the current chunk and start a new one
        if current_chunk["size"] + sent_len > max_chunk_size:
            chunks.append(current_chunk)

            current_chunk = {
                "size": 0,

                "tokens": [], 
                "flat_gold_labels": [] if labels else None,

                "sentences": [],
                "gold_labels": [] if labels else None,

                # metadata used for reconstruction
                "sent_lengths": [], 
                "sent_indices": []}

        # nested sentence storage
        current_chunk["sentences"].append(sent)

        # flat token storage
        current_chunk["tokens"].extend(sent)

        current_chunk["size"] += sent_len
        current_chunk["sent_lengths"].append(sent_len)
        current_chunk["sent_indices"].append(i)

        if labels:
            current_chunk["gold_labels"].append(labels[i])

            # flattened labels too
            current_chunk["flat_gold_labels"].extend(labels[i])

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
    print(f"Aligning gold tokens and LLM annotations...")
    
    gold_tokens = chunk["tokens"]
    aligned_labels = []

    pred_i = 0
    mismatch_count = 0

    for gold_i, gold_token in enumerate(gold_tokens):

        # no more predictions left
        if pred_i >= len(pred_pairs):
            aligned_labels.append("NO_PRED_LEFT")
            mismatch_count += 1
            continue

        pred_token, pred_label = pred_pairs[pred_i]

        # perfect match
        if pred_token == gold_token:
            aligned_labels.append(pred_label)
            pred_i += 1

        # if current gold token was skipped by model
        # and predicted token matches next gold

        elif (gold_i + 1 < len(gold_tokens) and pred_token == gold_tokens[gold_i + 1]):
            aligned_labels.append("GOLD_SKIPPED")
            mismatch_count += 1
            # do NOT advance pred_i
            # next loop will match this pred_token to next gold token

        # otherwise token mismatch
        else:
            aligned_labels.append("MISMATCH")
            pred_i += 1
            mismatch_count += 1

    print(f"Recovered chunk with {mismatch_count} mismatches out of {len(aligned_labels)} tokens.")
    return aligned_labels

    
# turn the string output from the model into a nested list
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
            label = "MISSING"
        
        # too many columns
        else:
            token = parts[0]
            label = "LONG_FORMAT"

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

            print(f"Sending prompt to {model_info['model_name']}, attempt {attempt+1}...")
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


# turn the aligned labels for the flatenned chunk and reconstruct the nested structure using chunk metadata
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


# iterate over the nested structure and write on the txt file dynamically
def write_iob2(sentences, labels, output_path):
    with open(output_path, "a", encoding="utf-8") as f:
        for sent_tokens, sent_labels in zip(sentences, labels): #iterating at the sentence level here

            for i, (token, label) in enumerate( #iterate inside each token-label pair
                zip(sent_tokens, sent_labels),
                start=1
            ):

                f.write(f"{i}\t{token}\t{label}\n")

            # blank line between sentences
            f.write("\n")


# for a list of sentences, tokenize inside each
def tokenize_sentences(sentences):
    tokenized_sentences = []

    pattern = r'\w+|[^\w\s]+' # match alphanumeric or match grouped special characters

    for sent in sentences:
        tokens = re.findall(pattern, sent)
        tokenized_sentences.append(tokens)

    return tokenized_sentences


# extract sentence list from dense text - tokenize contracts part 1
def extract_sentences(contract_path):
    with open(contract_path, "r") as f:
        text = f.read()

    nlp = spacy.load("en_core_web_sm")

    # Split on blank lines/new sections first
    blocks = re.split(r'\n', text)

    sentences = []

    for block in blocks:

        if not block:
            continue

        doc = nlp(block)

        for sent in doc.sents:
            sentences.append(sent.text.strip())

    # clean the sentence list
    fixed = []
    i = 0

    while i < len(sentences):

        current = sentences[i]

        # case 1 numbered section like 9.
        if re.match(r'^\d+\.$', current):

            if i + 1 < len(sentences):
                current += " " + sentences[i + 1]
                i += 1


        # case 2: don+t break clause markers like (iv)
        elif re.match(r'^\([a-zA-Z0-9ivx]+\)$', current):

            if i + 1 < len(sentences):
                current += " " + sentences[i + 1]
                i += 1


        # if the sentence starts with parenthesis and it is not a clause marker, it continues the prev sentence and they should go together
        elif (re.match(r'^\(', current)) and not (re.match(r'^\([a-zA-Z0-9ivx]+\)\s+', current)):

            # merge with previous sentence
            if fixed:
                fixed[-1] += " " + current
                i += 1
                continue

        # append final version of current
        fixed.append(current)

        # move to next sentence
        i += 1

    nested_sentences = tokenize_sentences(fixed)
    return nested_sentences


##### ----------------------------- MAIN SCRIPT -------------------------------------- #####

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="mistral")
    parser.add_argument("--quantized", action="store_true", help="Whether to load the model in 4-bit quantized mode for memory efficiency.")
    parser.add_argument("--mode", choices=["validate", "annotate"])
    parser.add_argument("--create_chunks", action="store_true")
    parser.add_argument("--chunk_file", default="chunks.json")
    parser.add_argument("--max_chunk_size", type=int, default=50, help="Maximum number of words to include in each chunk sent to the model for annotation.")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--input_data", default="annotation/FIN5_validation.txt")
    parser.add_argument("--output_dir", default="annotation/validation")
    parser.add_argument("--output_file", required=True)


    args = parser.parse_args()
    model_name = args.model_name.lower()
    quantized = args.quantized

    print(f"Loading model...\n")
    model = load_model(model_name, quantized)
    print(f"Model loaded.\n\n")
    
    # load data to annotate
    chunk_path = os.path.join("annotation/", args.chunk_file)
    
    if args.create_chunks:

        if args.mode == "validate":
            print(f"Extracting nested list of sentences from validation data...\n")
            sentences, true_labels = parse_iob2_file(args.input_data)
        else:
            print(f"Processing contract and extracting nested list of sentences...\n")
            sentences, true_labels = extract_sentences(args.input_data), None

        print(f"Extracting chunks of size ≤ {args.max_chunk_size} from sentences...\n")
        chunks = build_chunks(
            sentences,
            labels=true_labels,
            max_chunk_size=args.max_chunk_size
        )

        # since the output of creating chunks is deterministic, we can save it to reuse later
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)
        
        print(f"Chunks saved to {chunk_path}.\n\n")

    else:
        print(f"Loading chunks from {chunk_path}...\n")
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        sentences = []
        true_labels = [] if args.mode == "validate" else None

        for chunk in chunks:
            sentences.extend(chunk["sentences"])

            if args.mode == "validate":
                true_labels.extend(chunk["gold_labels"])

        print(f"Chunks and nested sentences loaded successfully.\n\n")

    
    print(f"Loading prompt template...\n")
    prompt_template = format_prompt("annotation/prompt_annotation.txt", "annotation/one_shot.txt")

    # create output dir from args
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    
    # clear previous file
    open(output_path, "w").close()

    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i+1}/{len(chunks)}")
        chunk_labels = annotate_chunk(prompt_template, chunk, model, args.temperature) # these are flat labels for the chunk
        nested_tokens, nested_labels = reconstruct_nested(chunk_labels, chunk)

        # save to file progressively
        write_iob2(nested_tokens, nested_labels, output_path) 

    print(f"\nResults saved to {output_path}")