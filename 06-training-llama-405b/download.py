import os
import torch
import transformers
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--skip-model", default=False, action="store_true")
args = parser.parse_args()

os.environ["HF_HOME"] = "/home/ubuntu/.cache/huggingface"

model_name = "meta-llama/Meta-Llama-3.1-405B"

print(f"Downloading {model_name} to $HF_HOME = {os.environ['HF_HOME']}.")

config = transformers.AutoConfig.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
if not args.skip_model:
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
