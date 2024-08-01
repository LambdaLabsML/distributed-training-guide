# Multi GPU across multiple nodes

```bash
OMP_NUM_THREADS=1 torchrun \
    --rdzv-id TODO \
    --rdzv-backend TODO \
    --rdzv-endpoint TODO \
    --nnodes 2 \
    --nproc-per-node gpu \
    train_llm.py \
    --experiment-name gpt2-openwebtext-A100 \
    --dataset-name Skylion007/openwebtext \
    --model-name openai-community/gpt2 \
    --batch-size 64
```

## How Multi Node works

It actually works in much the same way as the multi GPU. Since in the single node setting we have multiple processes, now we are just adding extra processes on different machines.

The main differences here to consider are:
1. How the nodes get in contact with each other (the `rdzv` arguments in the torchrun command)
2. Your code may need to use `local_rank` instead of `rank`. `rank` is between 0 and world_size, so if you have 2 machines, the second machine may have ranks 8-16. Local rank on the second machine will still be 0-8.

Error reporting/handling becomes extremely important with more than 1 node. Networking issues are very common, and there are some subtle things that you need to ensure are identical between the machines.

tl;dr: When going from single to multi node, ensuring environments are the same is the most important thing.

## Machine Setup

### Ensuring Environments are the same

### Ensuring System Dates & Times are the same

## Troubleshooting

How to set up your code/logging to make it easy to identify the cause?

### Networking Issues

### GPU Issues

### Hanging/Timeout issues

## Code Changes