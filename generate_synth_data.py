import argparse
import json
import os

import time
from datetime import timedelta

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--quantized", default=False, action="store_true", help="Whether to load the model in 4-bit quantized mode for memory efficiency.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", default="synthetic/")
    parser.add_argument("--output_file", default="synthetic_contract.txt")


    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.quantized:
        print("Loading model in 4-bit quantized mode...\n")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_4bit=True, #quantize the model for more memory efficiency
            trust_remote_code = False, #to ensure security when loading code from the model repository,
            device_map = "auto" #to automatically place model layers on available devices (e.g., GPU)
        )

    else:
        print("Loading model in full precision mode...\n")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype = torch.float16, #to make memory efficient
            trust_remote_code = False, #to ensure security when loading code from the model repository,
            device_map = "auto" #to automatically place model layers on available devices (e.g., GPU)
        )
        
    # disable training behavior
    model.eval()

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)

    prompt = f"""You are a synthetic legal document generator.

    Generate ONE realistic but fully fictional loan agreement contract.

    Rules:
    - All names, companies, and addresses must be fake.
    - The contract must be legally structured but NOT legally valid.
    - Include sections: parties, loan amount, interest rate, repayment terms, clauses, signatures.
    - Make it detailed and long (~{args.max_new_tokens} tokens worth of content).
    - Use formal legal language.

    Generate only the contract text.
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

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    
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