#!/bin/bash

EXPERIMENT_NAME=llama-405b

if [ ! -f ./hosts ]; then
    echo "ERROR: ./hosts file not found. Please add this file to this current directory."
    exit 1
fi

xargs \
    -a hosts \
    -I {} \
    ssh {} \
    tmux new-session -d -s torchrun-{} -c $(pwd) \
    -e HF_HOME=$(pwd)/../.cache \
    -e TORCHELASTIC_ERROR_FILE=../error.json \
    -e OMP_NUM_THREADS=1 \
    $(which python) -m torch.distributed.run \
    --rdzv-id ${EXPERIMENT_NAME} \
    --rdzv-backend c10d \
    --rdzv-endpoint $(head -n 1 hosts):5001 \
    --nnodes $(wc -l < hosts) \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ../logs \
    train_llm.py \
    --experiment-name ${EXPERIMENT_NAME} \
    --dataset-name Skylion007/openwebtext \
    --model-name meta-llama/Meta-Llama-3.1-405B \
    --batch-size 1 \
    --cpu-offload on \
    --bwd-prefetch off \
    --activations offload
