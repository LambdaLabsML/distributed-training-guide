import os
import torch
import transformers

os.environ["HF_HOME"] = "/home/ubuntu/.cache/huggingface"

model_name = "meta-llama/Meta-Llama-3.1-405B"

print(f"Downloading {model_name} to $HF_HOME = {os.environ['HF_HOME']}.")

config = transformers.AutoConfig.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
with torch.device("meta"):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
