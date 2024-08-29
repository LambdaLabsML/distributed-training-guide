# Job Launchers

Since it is quite cumbersome to manually SSH into every node and start a training job, there are various ways to launch distributed training jobs from a single node.

## mpirun

There are two main flavors of MPI implementation, OpenMPI and MPICH. Either of them will work and we will use the OpenMPI implementation in this blog. You need to install OpenMPI.

```bash
cd distributed-training-guide/04-job-launchers-mpirun
mpirun \
    -H <host 1>:<gpus on 1>,...,<host n>:<gpus on n> \
    -x MASTER_ADDR=<host 1> \
    -x MASTER_PORT=5001 \
    -x TORCHELASTIC_ERROR_FILE=../error.json \
    -x OMP_NUM_THREADS=1 \
    -bind-to none \
    -map-by slot \
    -wdir $(pwd) \
    -output-filename ../logs/mpi-multi-node \
    $(which python) train_llm.py \
    --experiment-name mpi-multi-node \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2 \
    --batch-size 64
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

```diff
@@ -6,6 +6,7 @@ import random
 import time
 from pathlib import Path
 import logging
+import os
 
 import torch
 from torch.utils.data import DataLoader
@@ -42,7 +43,11 @@ def main():
     numpy.random.seed(args.seed)
     random.seed(args.seed)
 
-    dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "mpi")
+    dist.init_process_group(
+        rank=int(os.environ["OMPI_COMM_WORLD_RANK"]),
+        world_size=int(os.environ["OMPI_COMM_WORLD_SIZE"]),
+        backend="nccl" if dist.is_nccl_available() else "mpi",
+    )
 
     rank = dist.get_rank()
     local_rank = rank % torch.cuda.device_count()
```
