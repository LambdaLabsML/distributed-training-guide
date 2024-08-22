# Job Launchers

Since it is quite cumbersome to manually SSH into every node and start a training job, there are various ways to launch distributed training jobs from a single node.

## Launching torchrun once per node with xargs, ssh, and tmux

Put your list of hostnames/IPs in a file called `hosts`
```
<hostname 1>
<hostname 2>
...
<hostname n>
```

Then we can use ssh to launch `torchrun` on each of the hosts.

```bash
xargs \
    -a hosts \
    -I {} \
    ssh {} \
    tmux new-session -d -s torchrun-{} -c $(pwd) \
    -e TORCHELASTIC_ERROR_FILE=../error.json \
    -e OMP_NUM_THREADS=1 \
    $(which python) -m torch.distributed.run \
    --rdzv-id multi-node-tmux \
    --rdzv-backend c10d \
    --rdzv-endpoint $(head -n 1 hosts):5001 \
    --nnodes $(wc -l < hosts) \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ../logs \
    train_llm.py \
    --experiment-name multi-node-tmux \
    --dataset-name Skylion007/openwebtext \
    --model-name openai-community/gpt2 \
    --batch-size 64
```

## Launching training script once per gpu with xargs, ssh, and tmux

Put your list of hostnames/IPs in a file called `gpus`. Each line should contain the hostname for a single gpu. If a single host has 8 GPUs, and you want to use all 8, that hostname should appear 8 separate times.

```
<hostname 1>
<hostname 2>
...
<hostname n>
```

```bash
cat -n gpus | xargs -n2 \
    bash -c 'ssh $1 \
    tmux new-session -d -s rank-$(($0 - 1)) -c $(pwd) \
    -e TORCHELASTIC_ERROR_FILE=../error.json \
    -e OMP_NUM_THREADS=1 \
    -e MASTER_ADDR=$(head -n 1 gpus) \
    -e MASTER_PORT=5001 \
    -e WORLD_SIZE=$(wc -l < gpus) \
    -e RANK=$(($0 - 1)) \
    $(which python) train_llm.py \
    --experiment-name multi-node-tmux \
    --dataset-name Skylion007/openwebtext \
    --model-name openai-community/gpt2 \
    --batch-size 64'
```
