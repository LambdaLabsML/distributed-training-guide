# Multi GPU on a single node

```bash
torchrun --standalone --nnodes 1 --nproc-per-node gpu train.py \
    --experiment-name gpt2-openwebtext-multi-gpu \
    --dataset-name Skylion007/openwebtext \
    --model-name openai-community/gpt2 \
    --batch-size 16
```
