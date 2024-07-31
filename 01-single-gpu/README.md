# Single GPU

This is the "standard" single gpu training script. It doesn't do anything with distributed, and aims to be as simple as possible.

## Command

```bash
python train.py \
    --experiment-name gpt2-openwebtext-single-gpu \
    --dataset-name Skylion007/openwebtext \
    --model-name openai-community/gpt2 \
    --batch-size 16
```

## Features

1. Resuming experiment (along with how to configure wandb to resume experiments).
2. Inner loop timing to monitor how long things are taking
3. 