import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype = torch.bfloat16,
    device_map = "auto",
    trust_remote_code = True
)

messages = [{
    "role":"user",
    "content": "Can you tell us 3 cities to visit in Turkey"
}]

tokenizer.apply_chat_template(messages, tokenize=False)

model_inputs = tokenizer.apply_chat_template(messages, return_tensors = "pt")