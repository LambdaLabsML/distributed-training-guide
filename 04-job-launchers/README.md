# Job Launchers

**NOTE: This chapter's code is identical to [chapter 3](../03-multi-node)'s code, so the command uses the training script from chapter 3.** If the job launcher requires code changes to work, the code changes will be called out.

Since it is quite cumbersome to manually SSH into every node and start a training job, there are various ways to launch distributed training jobs from a single node.

Quick jump:
- [Bash per node](#launching-torchrun-once-per-node)
- [Bash per gpu](#launching-script-once-per-gpu)
- [slurm](#slurm)
- [mpi](#mpirun)
- [deepspeed](#deepspeed)

## Bash Commands (xargs/ssh/tmux)

Since the main thing we need to do is spawn processes on other machines, we can combine a few bash tools together to achieve this. This approach is one of the most lightweight approaches for this, and makes it easy to edit the commands any way you want. While it takes a bit to understand how all the bash commands work together, they are generally applicable to other problems as well.

### Launching torchrun once per node

Put your list of hostnames/IPs in a file called `hosts`. Each line represents a single node that we will launch `torchrun` on.

```
<hostname 1>
<hostname 2>
...
<hostname n>
```

Then we can use ssh to launch `torchrun` on each of the hosts. This command is very similar to our previous bash command, except we are using `torchrun` (`python -m torch.distributed.run`) instead of just invoking our python script.

```bash
cd distributed-training-guide/04-job-launchers
xargs \
    -a hosts \
    -I {} \
    ssh {} \
    tmux new-session -d -s torchrun-{} -c $(pwd) \
    -e TORCHELASTIC_ERROR_FILE=../error.json \
    -e OMP_NUM_THREADS=1 \
    -e HF_HOME=../.cache \
    $(which python) -m torch.distributed.run \
    --rdzv-id multi-node-tmux \
    --rdzv-backend c10d \
    --rdzv-endpoint $(head -n 1 hosts):5001 \
    --nnodes $(wc -l < hosts) \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ../logs \
    ../03-multi-node/train_llm.py \
    --experiment-name multi-node-tmux \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2
```

### Launching script once per gpu

Put your list of hostnames in a file called `gpus`. Each line should contain the hostname for a single gpu. If a single host has 8 GPUs, and you want to use all 8, that hostname should appear 8 separate times.

```
<hostname 1>
<hostname 1>
...
<hostname n>
```

Then our command is:

```bash
cd distributed-training-guide/04-job-launchers
cat -n gpus | xargs -n2 \
    bash -c 'ssh $1 \
    tmux new-session -d -s rank-$(($0 - 1)) -c $(pwd) \
    -e TORCHELASTIC_ERROR_FILE=../error.json \
    -e OMP_NUM_THREADS=1 \
    -e HF_HOME=../.cache \
    -e MASTER_ADDR=$(head -n 1 gpus) \
    -e MASTER_PORT=5001 \
    -e WORLD_SIZE=$(wc -l < gpus) \
    -e RANK=$(($0 - 1)) \
    $(which python) ../03-multi-node/train_llm.py \
    --experiment-name multi-node-tmux \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2'
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

## slurm

slurm is a very popular job scheduling software often used with clusters.

Submit the training job using the provided `job.slurm` script:

```bash
cd distributed-training-guide/04-job-launchers
sbatch --nnodes 2 --gpus 16 --cpus-per-task 8 job.sbatch
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
export HF_HOME=../.cache

printenv

srun torchrun \
    --rdzv-id "slurm-${SLURM_JOBID}" \
    --rdzv-backend c10d \
    --rdzv-endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes ${SLURM_NNODES} \
    --nproc-per-node ${SLURM_GPUS_ON_NODE} \
    --redirects 3 \
    --log-dir ${SLURM_SUBMIT_DIR}/logs \
    ../03-multi-node/train_llm.py \
    --experiment-name gpt2-alpaca-slurm-$(date +%Y-%m-%dT%H-%M-%S) \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2
```

## mpirun

There are two main flavors of MPI implementation, OpenMPI and MPICH. Either of them will work and we will use the OpenMPI implementation in this blog. **You need to install OpenMPI**.

### Code Changes

Use MPI environment variables when initializing the process group:

```diff
-    dist.init_process_group()
+    dist.init_process_group(
+        rank=int(os.environ["OMPI_COMM_WORLD_RANK"]),
+        world_size=int(os.environ["OMPI_COMM_WORLD_SIZE"]),
+    )
```

### Command

```bash
cd distributed-training-guide/04-job-launchers
mpirun \
    -H <host 1>:<gpus on 1>,...,<host n>:<gpus on n> \
    -x MASTER_ADDR=<host 1> \
    -x MASTER_PORT=5001 \
    -x TORCHELASTIC_ERROR_FILE=../error.json \
    -x OMP_NUM_THREADS=1 \
    -x HF_HOME=../.cache \
    -bind-to none \
    -map-by slot \
    -wdir $(pwd) \
    -output-filename ../logs/mpi-multi-node \
    $(which python) train_llm.py \
    --experiment-name mpi-multi-node \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2
```

Arguments:
- `-H` specifies the hosts we want to launch on AND the number of processes per host
- `-x` sets up an environment variable in all the launched processes
- `-wdir` sets up the working directory for the launched processes
- `-bind-to none` specifies Open MPI to not bind a training process to a single CPU core (which would hurt performance).
- `-map-by slot` allows you to have a mixture of different NUMA configurations because the default behavior is to bind to the socket.

Notes:
- We have to specify `MASTER_ADDR` and `MASTER_PORT` for pytorch to know how to talk to each other
- In our code we have to pass the rank and world size based on the `$OMPI_COMM_WORLD_RANK` and `$OMPI_COMM_WORLD_SIZE` environment variables.
- We use `$(which python)` to get the absolute path of our python interpreter - if you are launch from a head node instead of a worker node, you'll need to change this.

## deepspeed

deepspeed is a distributed training library with many optimizations. We go into some of these optimizations in more detail in later chapters, but here we can just use the launcher included with it.

**NOTE: you do not have to integrate deepspeed into your training code to use the deepspeed launcher.**

Install: `pip install deepspeed`

### Code Changes

Add `--local_rank` to cli parsing:
```diff
     parser.add_argument("--log-freq", default=100, type=int)
     parser.add_argument("--ckpt-freq", default=500, type=int)
+    parser.add_argument("--local_rank", type=int, default=None)
     return parser
```

Use it when initializing local_rank:
```diff
-    local_rank = rank % torch.cuda.device_count()
+    local_rank = args.local_rank or (rank % torch.cuda.device_count())
```

### Command

```bash
cd distributed-training-guide/04-job-launchers
export HF_HOME=../.cache
export TORCHELASTIC_ERROR_FILE=../error.json
export OMP_NUM_THREADS=1
deepspeed \
    --include <ip of node 1>@<ip of node 2> \
    --enable_each_rank_log ../logs \
    train_llm.py \
    --experiment-name deepspeed-multi-node \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2
```
