# Job Launchers

Since it is quite cumbersome to manually SSH into every node and start a training job, there are various ways to launch distributed training jobs from a single node.

## slurm

slurm is a very popular job scheduling software often used with clusters.

Submit the training job using the provided `job.slurm` script:

```bash
sbatch --nnodes 2 --gpus 16 --cpus-per-task 8 job.slurm
```

By default slurm assigns 1 task per node, which is great for us because we will invoke torchrun once per node.

The command above requests a total of 16 gpus from 2 nodes total.

### The slurm file

This is mostly identical to our torchrun command that we have been using thus far, just with various settings controlled by slurm.

The command listed below will be run on each node (since we have specified `--ntasks-per-node=1`).

```bash
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
    --experiment-name gpt2-openwebtext-slurm-$(date +%Y-%m-%dT%H-%M-%S) \
    --dataset-name Skylion007/openwebtext \
    --model-name openai-community/gpt2 \
    --batch-size 64

```
