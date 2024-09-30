#!/bin/bash
# SBATCH --ntasks-per-node=1

source $(pwd)/../venv/bin/activate

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(expr 5000 + $(echo -n ${SLURM_JOBID} | tail -c 4))
export TORCHELASTIC_ERROR_FILE=./error-${SLURM_JOBID}-${SLURM_NODEID}.json
export OMP_NUM_THREADS=1

printenv

srun torchrun \
    --rdzv-id "slurm-${SLURM_JOBID}" \
    --rdzv-backend c10d \
    --rdzv-endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes ${SLURM_NNODES} \
    --nproc-per-node ${SLURM_GPUS_ON_NODE} \
    --redirects 3 \
    --log-dir ${SLURM_SUBMIT_DIR}/logs \
    train_llm.py \
    --experiment-name gpt2-alpaca-slurm-$(date +%Y-%m-%dT%H-%M-%S) \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2
