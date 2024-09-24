import os
import torch
import transformers

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.curdir, "..", ".cache"))

model_name = "meta-llama/Meta-Llama-3.1-405B"

print(
    f"Downloading {model_name} to $HF_HOME = {os.environ['HF_HOME']}. Please ensure this is a shared network drive accessible to all nodes."
)

with torch.device("meta"):
    config = transformers.AutoConfig.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
