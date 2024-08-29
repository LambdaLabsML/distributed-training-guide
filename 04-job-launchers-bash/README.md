# Job Launchers

Since it is quite cumbersome to manually SSH into every node and start a training job, there are various ways to launch distributed training jobs from a single node.

## Bash Commands

Since the main thing we need to do is spawn processes on other machines, we can combine a few bash tools together to achieve this. This approach is one of the most lightweight approaches for this, and makes it easy to edit the commands any way you want. While it takes a bit to understand how all the bash commands work together, they are generally applicable to other problems as well.

### Launching training script once per gpu with xargs, ssh, and tmux

Put your list of hostnames in a file called `gpus`. Each line should contain the hostname for a single gpu. If a single host has 8 GPUs, and you want to use all 8, that hostname should appear 8 separate times.

```
<hostname 1>
<hostname 1>
...
<hostname n>
```

Then our command is:

```bash
cd distributed-training-guide/04-job-launchers-bash
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

Here is how this command works:

1. `cat -n` will "enumerate" the lines in the file so we can get the rank of each device to use
2. We pipe these into `xargs -n2` to split these values into the `$0` and `$1` variables.
3. It's important to call `bash -c '<command>'` so we can access the `$0` and `$1` variables.
4. Our command is made up of calling a command on a remote machine via the `ssh $1 <command>` command.
5. Our remote command will be using `tmux new-session` to spawn a new process that we can attach to for each rank.
    1. `-d` means detached, so we can spawn it without blocking
    2. `-s rank-$(($0 - 1))` means the name of the session will be `rank-{i}`
    3. `-c $(pwd)` means every process will have the working directory that we launch this command from
    4. `-e <env variable name>=<value>` will set up an environment variable for the process we launch using tmux

We need a couple of environment variables to make `dist.init_process_group()` work:
1. `MASTER_ADDR`/`MASTER_PORT` is equivalent to the `rdzv` arguments with torchrun, they let each process connect to a single address. Since our `gpus` file contains a list of filenames, we just arbitrarily use the first one (`head -n 1 gpus`) as our master address
2. `WORLD_SIZE` which we can just count the lines in our `gpus` file (`wc -l < gpus`) since each line represents a single GPU
3. `RANK` which we have from our enumerated cat command - though we have to subtract 1 since `cat` enumerates starting at 1 - `$(($0 - 1))`

From there on we just paste our normal python command, note that we use `$(which python)` to get the absolute path to whatever interpreter executable we are using. 

### Launching torchrun once per node with xargs, ssh, and tmux

Put your list of hostnames/IPs in a file called `hosts`. Each line represents a single node that we will launch `torchrun` on.

```
<hostname 1>
<hostname 2>
...
<hostname n>
```

Then we can use ssh to launch `torchrun` on each of the hosts. This command is very similar to our previous bash command, except we are using `torchrun` (`python -m torch.distributed.run`) instead of just invoking our python script.

```bash
cd distributed-training-guide/04-job-launchers-bash
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