import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype = torch.float16, #to make memory efficient
    trust_remote_code = False #to ensure security when loading code from the model repository 
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prompt = """
You are a synthetic legal document generator.

Generate ONE realistic but fully fictional loan agreement contract.

Rules:
- All names, companies, and addresses must be fake.
- The contract must be legally structured but NOT legally valid.
- Include sections: parties, loan amount, interest rate, repayment terms, clauses, signatures.
- Make it detailed and long (~2000 tokens worth of content).
- Use formal legal language.

Generate only the contract text.
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate text
output = model.generate(
    **inputs,
    max_new_tokens=2000,
    temperature=0.8,
    top_p=0.95,
    do_sample=True
)

text = tokenizer.decode(output[0], skip_special_tokens=True)

# Save to file
with open("synthetic_contract.txt", "w", encoding="utf-8") as f:
    f.write(text)