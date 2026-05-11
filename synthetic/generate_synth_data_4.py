########### IMPORTS #####################
import argparse
import os
import json
import re
from faker import Faker
fake = Faker()
import time
from datetime import timedelta

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

#################### ---------HELPER FUNCTIONS--------- ################################

# general function to generate text given a prompt, model and tokenizer
def generate(model, tokenizer, model_name, prompt, role, max_new_tokens, use_chat_template, temperature=0.6):
    # format prompt according to model requirements
    if use_chat_template:
        messages = [
            {"role": "system", "content": f"You are a legal document {role}."},
            {"role": "user", "content": prompt}
        ]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    
    # prepare to track generation time        
    start_time = time.perf_counter()  

    # generate text
    with torch.no_grad():  #telling not to track gradients since we're only generating text, not training
        output = model.generate(
            **inputs, #tokenized prompt inputs
            max_new_tokens=max_new_tokens, #set max tokens to generate
            temperature=temperature, #control randomness
            top_p=0.8, #increase word selection diversity
            do_sample=True, #used with top_p
            use_cache=True, #speed up the process by caching past key values
            eos_token_id=tokenizer.eos_token_id #stop generation when EOS token is produced
        )

    input_len = inputs["input_ids"].shape[-1] #number of tokens in the input prompt, to know where the generated tokens start
    generated_tokens = output[0][input_len:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    #metrics for generation time and GPU memory usage
    end_time = time.perf_counter()
    elapsed_time = timedelta(seconds=end_time - start_time)

    print(f"Generation time: {elapsed_time}")

    return text



# generate the entire contract in one go using the full prompt
def generate_full(model, tokenizer, model_name, args, use_chat_template, temperature=0.6):
    # load prompt template from file
    with open("prompt-full.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(
        max_new_tokens=args.max_new_tokens
    )

    # get ready to track GPU stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("Generating full contract...\n")
    output = generate(model, tokenizer, model_name, prompt, role="legal text generator", max_new_tokens=args.max_new_tokens, use_chat_template=use_chat_template, temperature=temperature)

    if torch.cuda.is_available():
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        gpu_mem_reserved = torch.cuda.max_memory_reserved() / (1024**3)

        print(f"GPU memory allocated: {gpu_mem_alloc:.2f} GB")
        print(f"GPU memory reserved:  {gpu_mem_reserved:.2f} GB\n")

    print("Full contract generation complete.\n")
    return output



# generate the contract section by section using the plan and sections generated from the model
def generate_sectioned(model, tokenizer, model_name, args, use_chat_template, plan_temperature=0.8, section_temperature=0.6):
    print("Generating plan...\n")

    with open("prompt-plan-3.txt", "r", encoding="utf-8") as f:
        plan_prompt = f.read()

    # prepare to track GPU stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    jurisdiction = f"{fake.country()}"
    agreement_date = f"{fake.date()}"
    interest_rate = f"{fake.random_int(2, 12)}%"
    fake_contact_lender = {
        'address': fake.address().replace('\n', ', '),
        'email': fake.email(),
        'phone': fake.phone_number()
    }
    fake_contact_borrower = {
        'address': fake.address().replace('\n', ', '),
        'email': fake.email(),
        'phone': fake.phone_number()
    }

    context_block = f"""
    Use the following information to generate plan:

    Lender: generate fake organization name (Contact: {fake_contact_lender['address']}, {fake_contact_lender['email']}, {fake_contact_lender['phone']})
    Borrower: generate fake company name (Contact: {fake_contact_borrower['address']}, {fake_contact_borrower['email']}, {fake_contact_borrower['phone']})
    Jurisdiction: {jurisdiction}
    Agreement Date: {agreement_date}
    Interest Rate: {interest_rate}
    """
    plan_prompt = f"""
        You are a legal document planner.
        
        Design a structured plan for a realistic corporate loan agreement between organizations.

        {context_block}

        {plan_prompt}
    """


    # ask the llm to generate a valid plan in json format, if not successful retry 3 times before giving up
    for attempt in range(3):
        try:
            raw_output = generate(model, tokenizer, model_name, plan_prompt, role="planner", max_new_tokens=args.max_new_tokens, use_chat_template=use_chat_template, temperature=plan_temperature)
            print(f"[Attempt {attempt+1}] Raw plan output:\n{raw_output[:500]}\n")
            
            raw_output = extract_json(raw_output)
            plan_output = json.loads(raw_output)

            
            # structure validtion checks
            if not isinstance(plan_output, dict):
                raise ValueError("Plan is not a JSON object")

            required_keys = ["plan", "sections", "parties", "loan_purpose", "interest_rate", "replayment_terms"]
            if not all(k in plan_output for k in required_keys):
                raise ValueError("Missing required keys")
            
            if not isinstance(plan_output["sections"], list) or len(plan_output["sections"]) == 0:
                raise ValueError("Invalid sections list")

            if not isinstance(plan_output["parties"], list) or len(plan_output["parties"]) == 0:
                raise ValueError("Invalid parties list")

            print(f"Valid plan generated (attempt {attempt+1})\n")
            break
        except Exception as e:
            print(f"[Attempt {attempt+1}] Failed: {e}\n")
            continue

    else:
        raise ValueError("Failed to generate valid plan after retries")

    plan = plan_output["plan"]

    sections = plan_output["sections"]
    section_list = "\n".join([f"- {i+1}. {s['title']}" for i,s in enumerate(sections)])
    
    parties = plan_output["parties"]
    parties_list ="\n".join([f"- {p['name']} ({p['role']})" for p in parties])

    loan_purpose = plan_output.get("loan_purpose", "Not specified")
    replayment_terms = plan_output.get("replayment_terms", "Not specified")
    contact_information = plan_output.get("contact_information", "Not specified")

    print("Plan generation complete.\n")
    print(f"Plan description:\n{plan}\n")
    print(f"Sections:\n{section_list}\n")
    print(f"Parties:\n{parties_list}\n")
    print(f"Jurisdiction: {jurisdiction}")
    print(f"Loan Purpose: {loan_purpose}")
    print(f"Agreement Date: {agreement_date}")
    print(f"Interest Rate: {interest_rate}")
    print(f"Replayment Terms: {replayment_terms}")
    print(f"Contact Information: {contact_information}\n")

    full_text = ""


    print("Generating contract in sections...\n")
    with open("prompt-sectioned-3.txt", "r", encoding="utf-8") as f:
        section_template = f.read()

    for i, section in enumerate(sections):
        title = f"{i+1}. {section.get('title', 'Untitled Section')}"
        summary = section.get("summary", "")

        print(f"[{i+1}/{len(sections)}] Generating: {title}")

        section_prompt = section_template.format(
            global_plan=plan,
            parties=parties_list,
            jurisdiction=jurisdiction,
            loan_purpose=loan_purpose,
            agreement_date=agreement_date,
            interest_rate=interest_rate,
            replayment_terms=replayment_terms,
            contact_information=contact_information,
            sections_list=section_list,
            section_title=title,
            section_description=summary,
            prev_text=full_text[-500:]
        )

        section_text = generate(model, tokenizer, model_name, section_prompt, role="generator", max_new_tokens=args.max_new_tokens, use_chat_template=use_chat_template, temperature=section_temperature)

        # Remove repeated section title if present at the start of the generated text
        section_text_stripped = section_text.strip()
        # Remove both exact title and title without numbering (e.g., '1. Loan Terms' and 'Loan Terms')
        if section_text_stripped.startswith(title):
            section_text_stripped = section_text_stripped[len(title):].lstrip(': .\n')
        # Also check for just the section name (without number)
        section_name = section.get('title', '').strip()
        if section_name and section_text_stripped.startswith(section_name):
            section_text_stripped = section_text_stripped[len(section_name):].lstrip(': .\n')

        full_text += f"\n\n{title}\n{section_text_stripped}"

        if torch.cuda.is_available() and (i % 2 == 0):
            torch.cuda.empty_cache() # clear GPU memory every second section to avoid OOM errors

    if torch.cuda.is_available():
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            gpu_mem_reserved = torch.cuda.max_memory_reserved() / (1024**3)

            print(f"GPU memory allocated: {gpu_mem_alloc:.2f} GB")
            print(f"GPU memory reserved:  {gpu_mem_reserved:.2f} GB\n")

    return full_text


# helper function to extract the JSON from the text generated by the LLM
def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    return match.group(0)


##### ----------------------------- MAIN SCRIPT -------------------------------------- #####

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="mistral")
    parser.add_argument("--quantized", action="store_true", help="Whether to load the model in 4-bit quantized mode for memory efficiency.")
    parser.add_argument("--sectioned", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=2000)
    parser.add_argument("--output_dir", default="contracts/")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--plan_temperature", type=float, default=0.9)
    parser.add_argument("--section_temperature", type=float, default=0.6)


    args = parser.parse_args()
    if args.model_name.lower() == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        use_chat_template = False
    elif args.model_name.lower() == "llama":
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        use_chat_template = True
    elif args.model_name.lower() == "qwen":
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        use_chat_template = True
    elif args.model_name.lower() == "deepseek":
        model_name = "deepseek-ai/deepseek-llm-7b-chat"
        use_chat_template = True
    elif args.model_name.lower() == "gemma":
        model_name = "google/gemma-2b-it"
        use_chat_template = False
    elif args.model_name.lower() == "test":
        model_name = "sshleifer/tiny-gpt2"
        use_chat_template = False
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}. Supported options are: mistral, llama, qwen, deepseek, gemma.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
            device_map="auto", #to automatically place model layers on available devices (e.g., GPU)
            trust_remote_code=True, # since we are only loading well-known models and some require custom code execution
            use_safetensors=True
        )

    else:
        print("Loading model in full precision mode...\n")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype = torch.float16, #to make memory efficient
            device_map="auto", #to automatically place model layers on available devices (e.g., GPU)
            trust_remote_code=True, # since we are only loading well-known models and some require custom code execution
            use_safetensors=True 
        )
        
    # disable training behavior
    model.eval()

    if args.sectioned:
        text = generate_sectioned(
            model, tokenizer, model_name, args, use_chat_template,
            plan_temperature=args.plan_temperature, section_temperature=args.section_temperature
        )
    else:
        text = generate_full(model, tokenizer, model_name, args, use_chat_template, temperature=args.section_temperature)
    
    # create output dir from args
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Synthetic contract saved to {output_path}")