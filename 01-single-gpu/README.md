# Single GPU

This is the "standard" single gpu training script. It doesn't do anything with distributed, and aims to be as simple as possible.

## Command

```bash
python train_llm.py \
    --experiment-name gpt2-openwebtext-single-gpu \
    --dataset-name Skylion007/openwebtext \
    --model-name openai-community/gpt2 \
    --batch-size 64
```

## Features

TODO