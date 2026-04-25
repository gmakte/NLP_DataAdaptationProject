import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype = torch.bfloat16, #to make memory efficient
    trust_remote_code = False #to ensure security when loading code from the model repository 
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

messages = [{
    "role":"user",
    "content": "Can you tell us 3 cities to visit in Turkey"
}]

tokenizer.apply_chat_template(messages, tokenize=False)

model_inputs = tokenizer.apply_chat_template(messages, return_tensors = "pt")

generated_ids = model.generate(
    model_inputs,
    max_new_tokens = 500, #to limit the length of generated response
    do_sample = True,
)

decoded = tokenizer.batch_decode(generated_ids)

print(decoded[0])
