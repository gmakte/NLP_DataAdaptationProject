import argparse
import json
import os

import time
from datetime import timedelta

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--quantized", default=False, action="store_true", help="Whether to load the model in 4-bit quantized mode for memory efficiency.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", default="synthetic/")
    parser.add_argument("--output_file", default="synthetic_contract.txt")
    parser.add_argument("--example_start", default=2, type=int)
    parser.add_argument("--example_end", default=5, type=int)


    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.quantized:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        print("Loading model in 4-bit quantized mode...\n")

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

    else:
        print("Loading model in full precision mode...\n")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype = torch.float16, #to make memory efficient
            trust_remote_code = False, #to ensure security when loading code from the model repository,
            device_map = "auto" #to automatically place model layers on available devices (e.g., GPU)
        )
        
    # disable training behavior
    model.eval()

    # examples = []
    # with open("data/FIN5_sentences.txt", "r", encoding="utf-8") as f:
    #     i=1
        
    #     for line in f:
    #         if i >= args.example_start and i <= args.example_end:
    #             examples.append(line.strip())
    #         i += 1
    
    # example_text ='\n'.join(examples)

    prompt = f"""You are a legal document generator.

    Generate a realistic, fully specified loan agreement.

    Requirements:

    - Start with a formal opening paragraph introducing:
    - the agreement type
    - a specific date (e.g., "January 14, 2021")
    - the parties, defined with roles in parentheses (e.g., "Bank", "Borrower")

    - Use numbered sections (e.g., 1, 2, 3)
    - Include at least the following sections:
        - Loan Terms
        - Interest and Fees
        - Default and Remedies
        - Signatures

    You may add additional sections, but:
    - Maintain consistent numbering (no skipped or repeated numbers)
    - Use clear section headings
    - Keep formatting consistent across the entire contract
    - Ensure the contract is internally coherent and consistent:
        - Use the same party names and roles throughout
        - Ensure all clauses align with previously defined terms
        - Do not introduce contradictions between sections
        - Keep monetary amounts, dates, and obligations consistent across the document

    Style:
    - Use formal legal language
    - Use long, structured sentences
    - Use phrases like "WHEREAS" and "NOW, THEREFORE" where appropriate

    Constraints:
    - Use fully fictional names, companies, and addresses
    - Do NOT use placeholders like "____", "________", or blanks
    - Always provide concrete values for:
    - dates
        - monetary amounts
        - interest rates
        - names and addresses
        - signatures (include names and titles, no empty lines)
    - Do NOT repeat the instructions or include separators like "-----"
    - Avoid repetition in the contract content
    - Keep length around {args.max_new_tokens} tokens

    Output only the contract text.
    """

    print("Generating synthetic data...\n")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # prepare to track generation time and GPU memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.perf_counter()    

    # Generate text
    with torch.no_grad(): #telling not to track gradients since we're only generating text, not training
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            use_cache=True
        )

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    #metrics for generation time and GPU memory usage
    end_time = time.perf_counter()
    elapsed_time = timedelta(seconds=end_time - start_time)

    print(f"Synthetic data generation complete.\nGeneration time: {elapsed_time}\n")

    if torch.cuda.is_available():
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        gpu_mem_reserved = torch.cuda.max_memory_reserved() / (1024**3)

        print(f"GPU memory allocated: {gpu_mem_alloc:.2f} GB")
        print(f"GPU memory reserved:  {gpu_mem_reserved:.2f} GB\n")
    
    # create output dir from args
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Synthetic contract saved to {output_path}")