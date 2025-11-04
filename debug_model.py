import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.mac_config import mac_config

model_path = "models/retail-assistant"
device = mac_config.get_device()

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

base_model_name = "microsoft/phi-2"
with mac_config.mutex_guard():
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True
    )

model = PeftModel.from_pretrained(base_model, model_path)
model = model.to(device)
model.eval()

print(f"Tokenizer pad_token: {tokenizer.pad_token}")
print(f"Tokenizer eos_token: {tokenizer.eos_token}")
print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")

# Test with a training example
prompt = """### Instruction: You are a helpful Macy's virtual shopping assistant. Help the customer find products and provide detailed information.
### Input: What's the price of Coach sunglasses?
### Response:"""

print(f"\nPrompt:\n{prompt}\n")

inputs = tokenizer(prompt, return_tensors="pt").to(device)
print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Input tokens: {inputs['input_ids'][0][:20]}")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id  # Use eos as pad
    )

full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\nFull response:\n{full_response}\n")

clean_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nClean response:\n{clean_response}\n")

if "### Response:" in clean_response:
    answer = clean_response.split("### Response:")[-1].strip()
    if "###" in answer:
        answer = answer.split("###")[0].strip()
    print(f"\nExtracted answer:\n{answer}")
