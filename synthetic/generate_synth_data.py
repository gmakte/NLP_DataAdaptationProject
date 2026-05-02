import argparse
import os

import time
from datetime import timedelta

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="mistral")
    parser.add_argument("--quantized", default=False, action="store_true", help="Whether to load the model in 4-bit quantized mode for memory efficiency.")
    parser.add_argument("--min_new_tokens", type=int, default=2000)
    parser.add_argument("--output_dir", default="contracts/")
    parser.add_argument("--output_file", default="synthetic_contract.txt")


    args = parser.parse_args()
    if args.model_name.lower() == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    elif args.model.lower() == "llama":
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif args.model.lower() == "qwen":
        model_name = "Qwen/Qwen2.5-7B-Instruct"
    elif args.model.lower() == "deepseek":
        model_name = "deepseek-ai/deepseek-llm-7b-chat"
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}. Supported options are: mistral, llama, qwen, deepseek.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.quantized:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        print("Loading model in 4-bit quantized mode...\n")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

    else:
        print("Loading model in full precision mode...\n")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype = torch.float16, #to make memory efficient
            trust_remote_code = True, # since we are only loading well-known models and some require custom code execution
            device_map = "auto" #to automatically place model layers on available devices (e.g., GPU)
        )
        
    # disable training behavior
    model.eval()

    # load prompt template from file
    with open("prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(
        min_new_tokens=args.min_new_tokens
    )

    # format prompt according to model requirements
    use_chat_template = any(x in model_name for x in ["llama", "qwen", "deepseek"])

    if use_chat_template:
        messages = [
            {"role": "system", "content": "You are a legal document generator."},
            {"role": "user", "content": prompt}
        ]

        print("Generating synthetic data...\n")
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    else:
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
            min_new_tokens=args.min_new_tokens,
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