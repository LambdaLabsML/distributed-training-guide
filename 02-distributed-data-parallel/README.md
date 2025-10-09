# Multi GPU on a single node

**NOTE: This chapter's code builds off of [chapter 1](../01-single-gpu).**

Single node command:

```bash
cd distributed-training-guide/02-distributed-data-parallel
export TORCHELASTIC_ERROR_FILE=../error.json
export OMP_NUM_THREADS=1
torchrun \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ../logs \
    train_llm.py \
    -d tatsu-lab/alpaca \
    -m openai-community/gpt2
```

For multi node, see our [chapter on job launchers](../03-job-launchers/).

Quick jump:
- [How Distributed Training works](#how-distributed-training-works)
- [Using torchrun](#using-torchrun-instead-of-python)
- [Code Changes](#code-changes)
    - [Calling dist.init_process_group() and torch.cuda.set_device()](#calling-distinit_process_group-and-torchcudaset_device)
    - [Including rank in logging statements](#including-rank-in-logging-statements)
    - [DistributedDataParallel (DDP)](#using-distributeddataparallel)
    - [DistributedSampler](#using-distributedsampler)
    - I/O related guards
        - [Downloading model/data in rank 0 first](#downloading-model--data-in-rank-0-first)
        - [Interacting with file system on rank 0 only](#only-creating-experiment-directory-on-rank-0)
        - [wandb on rank 0 only](#wandb-runs-on-rank-0)
        - [Checkpoints from rank 0 only](#save-checkpoint-on-rank-0)
- [Optimizing memory - Zero Redundancy](#optimizing-memory---zero-redundancy-optimizer)
- [How multi node works](#how-multi-node-works)
- [Shared storage - Managing your python virtual environment across nodes](#shared-storage---managing-your-python-virtual-environment-across-nodes)
- [Shared storage - Mangaging your dataset/model checkpoints across nodes](#shared-storage---mangaging-your-datasetmodel-checkpoints-across-nodes)
- [`$HF_HOME` - The downloaded Model/Dataset directory](#hf_home---the-downloaded-modeldataset-directory)

## Dictionary

- `world size`: the total number of participating gpus
- `rank` the global unique id of this worker (from `0` up to and including `world_size - 1`)
- `local_rank` the rank local to this machine (from `0` up to and including `torch.cuda.device_count() - 1`)

## How Distributed Training works

Before we get into the changes required to do distributed training, let's think a little bit. If you've ever done parallel computations before, you know one way to achieve parallelization is to simply split your workload over all your cores. This is really useful if your task is relatively the same for all of the things you want to process. In fact, this is how python's multiprocessing.Pool.map object works.

Well distributed training with a GPU actually works the same way - we are splitting our workload (which is the batches from our dataset) over multiple GPUs. However we have an additional problem: how do we ensure that the model on all of our GPUs is the same?

We can actually achieve this in a very clever way. For sake of simplicity let's assume:
1. Our model and optimizer fully fit on every GPU
2. We initialize our model the exact same way on all of our GPUs
3. Our optimizer has the exact same settings on all of our GPUs

Now let's focus on our training loop. The canonical one in pytorch is:

```python
loss = model(**batch) # 1. Forward pass asynchronously
optimizer.zero_grad() # 2. Reset gradients asynchronously
loss.backward()       # 3. calculates gradients asynchronously
optimizer.step()      # 4. synchronize gradients & update weights
```

The first 3 lines of the above can all be done asychronously. `loss.backward()` will compute gradients in each of our training processes. The clever bit is that `optimizer.step()` will synchronize the gradients across all processes before actually updating the model parameters.

Here is a high level depiction of how a single step of training works when using this data parallel technique:

<img width="800" alt="image" src="https://github.com/user-attachments/assets/e228d298-30dc-4c3d-a550-e88ab55ddd93">

So to be explicit: **`optimizer.step()` is a synchronization point across ALL processes**.

So how does pytorch achieve this?

1. [Running N copies of our training script with torchrun.](#using-torchrun-instead-of-python)
2. [Splitting data across our workers with DistributedSampler](#using-distributedsampler)
3. [Synchronizing our gradients with DistributedDataParallel](#using-distributeddataparallel)

## Using `torchrun` instead of `python`

When we use `torchrun` to launch a distributed training job, what's happening is that it is **launch N separate processes** (where N is the number of gpus you specify in `--nproc-per-node`), all running your same training script:

```
> torchrun --nproc-per-node 3 train_llm.py ...
Launches subproc `$RANK=0 $WORLD_SIZE=3 train_llm.py ...`
Launches subproc `$RANK=1 $WORLD_SIZE=3 train_llm.py ...`
Launches subproc `$RANK=2 $WORLD_SIZE=3 train_llm.py ...`
```

It will also set up some synchronization between the processes. Then each of the processes is running the same training code and needs to synchronize at various points. Each of the processes gets an id (the `rank`), which will tell it what device to use.

When running on multiple nodes, you need to run torchrun on every machine, but other than that it works exactly the same. See our [job launchers chapter](../03-job-launchers/) for how to do this.

Here are some of the common CLI arguments to torchrun used throughout this guide:
- `--standalone` argument is used when only running on a single node.
- `--nnodes` is the number of nodes we are using, in this case 1, but once we go to multiple nodes, this will be > 1.
- `--nproc-per-node` is the number of processes. `gpu` means to use all available GPUs.
- `--redirects 3` redirects the stdout & stderr into files
- `--log-dir ../logs` configures the log directory

#### TORCHELASTIC_ERROR_FILE

**Very important to include this for debugging!**

When one of the workers (including a thread from a worker process) has an error, torchrun will save the error to the filepath controlled by this environment variable.

You also need to add a `@record` (imported `from torch.distributed.elastic.multiprocessing.errors import record`) annotation to your main function:

```diff
+@record
 def main():
```

#### OMP_NUM_THREADS

pytorch by default tries to take advantage of all the cores available when doing computations, even when you are on the GPU. Since we have multiple processes running pytorch, if we didn't set `OMP_NUM_THREADS` to something else, all of them would try to use all available cores.

You can manually check how many available cores there are and then split them accordingly. E.g. if there were 32 cores on a machine and 8 GPUs, you could set OMP_NUM_THREADS to 4.


## Code Changes

### Calling `dist.init_process_group()` and `torch.cuda.set_device()`

You are required to call both of these before calling other dist apis.

`dist.init_process_group()` will block until `WORLD_SIZE` processes have called it.

```diff
+from torch import distributed as dist

 def main():
     parser = _get_parser()
     args = parser.parse_args()

+   rank = int(os.getenv("RANK", "0"))
+   local_rank = rank % torch.cuda.device_count()
+   world_size = int(os.getenv("WORLD_SIZE", "1"))

-   device = torch.device(f"cuda") 
+   device = torch.device(f"cuda:{local_rank}")
+   torch.cuda.set_device(device)

+   dist.init_process_group(rank=rank, world_size=world_size, device_id=device)
```

If you don't call torch.cuda.set_device, processes may not be using the correct CUDA device.

### Including rank in logging statements

This is a helpful thing to do to handle all the processes outputting to the same file, or even when you're browsing a single log file it's useful to have this on every log statement:

```diff
 logging.basicConfig(
-    format=f"[%(asctime)s] %(levelname)s:%(message)s",
+    format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
     level=logging.INFO,
 )
```

### Using DistributedDataParallel

```diff
+from torch.nn.parallel import DistributedDataParallel

 with device: 
     model = AutoModelForCausalLM.from_config(config, dtype=dtype)

+model = DistributedDataParallel(model, device_ids=[local_rank])
```

Funnily enough you might assume that the DDP module splits batches across processes, but that is not what it does at all!

This is a model wrapper class that ensures **gradients are synchronized before calling optimizer.step()**. I encourage you to read the documentation for this, it's very informative: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html.

This class also ensures that *model parameters are equal when you construct it*!

It achieves all of this through some [very special model hooks](https://github.com/pytorch/pytorch/blob/v2.4.1/torch/nn/parallel/distributed.py#L939) to sum all the gradients from all the ranks on all the ranks together:

```python
# NOTE: internal pytorch code found at https://github.com/pytorch/pytorch/blob/v2.4.1/torch/nn/parallel/distributed.py#L939
gradient = param.grad / self.process_group.size()
gradient = fcol.all_reduce(gradient, "sum", self.process_group)
```

### Using DistributedSampler

In our normal training script we use a `torch.utils.data.DataLoader` to batch our data. One of the arguments to DataLoader is a `sampler`, which basically samples items from the dataset when constructing the batches. You can think of the sampler as doing:

```python
def simple_sampler():
    worker_len = len(dataset)
    return random.choice(range(worker_len))
```

The clever thing that the DistributedSampler does is it partitions the length of the dataset across each of our workers. You don't even have to partition the actual dataset - it just chooses the integers that it returns from a specific subset of the dataset:

```python
def distributed_sampler():
    worker_len = len(dataset) // dist.get_world_size()
    return dist.get_rank() * worker_len + random.choice(range(worker_len))
```

Our code changes are very minimal for this to work!

```diff
+from torch.utils.data.distributed import DistributedSampler

 dataloader = DataLoader(
     train_data,
     batch_size=args.batch_size,
-    shuffle=True,
-    drop_last=True,
     collate_fn=default_data_collator,
+    sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
 )
```

You also need to call [DistributedSampler.set_epoch](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler). Here's the quote from the pytorch doc on this:

```diff
 for state["epoch"] in range(state["epoch"], args.num_epochs):
+    dataloader.sampler.set_epoch(state["epoch"])
     batches = iter(dataloader)
```

> In distributed mode, calling the set_epoch() method at the beginning of each epoch before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used.

### Downloading model & data in rank 0 first

This is mainly necessary because loading our data may download/preprocess some data and write to disk.

If we didn't do rank 0 first, all of our ranks may try to download the data at once, which will slow everything down.

**NOTE: A good best practice is to have your data already downloaded & preprocessed into a shared network drive**

We can add a simple context manager to do this:

```python
@contextmanager
def rank0_first():
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()
```

Downloading model weights & tokenizer:

```diff
+with rank0_first():
     config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
     with device:
         model = AutoModelForCausalLM.from_config(config, dtype=dtype)
```

Downloading data:

```diff
+with rank0_first():
     train_data = _load_and_preprocess_data(args, tokenizer, config)
```

### Only creating experiment directory on rank 0

Note the `dist.barrier()` calls before and after we create the directory. **These are very important!**

Since we check to see if the experiment directory already exists right before creating the experiment directory, we need to ensure that **all processes have checked for its existence**. So the first `dist.barrier()` call ensures that all workers have already checked the existence of that. Then and only then can we create the directory on rank 0.

```diff
+if rank == 0:
     exp_dir.mkdir(parents=True, exist_ok=True)
+dist.barrier()
```

### Save checkpoint on rank 0

We only want one of our ranks to save a checkpoint. Otherwise the ranks might write to the same file and corrupt each other.

```diff
 if state["global_step"] % args.ckpt_freq == 0:
+    if rank == 0:
         torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
         torch.save(model.state_dict(), exp_dir / "model.pt")
         torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
         with open(exp_dir / "state.json", "w") as fp:
              json.dump(state, fp)
+    dist.barrier()
```

## Optimizing memory - Zero Redundancy Optimizer

DDP stores the entire model and optimizer on every single GPU. This is especially wasteful regarding the optimizer. Thankfully we have [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) which we can easily add to reduce memory usage:

```diff
+ from torch.distributed.optim import ZeroRedundancyOptimizer

-optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
+optimizer = ZeroRedundancyOptimizer(
+    model.parameters(), optimizer_class=torch.optim.AdamW, lr=args.lr
+)
```

Unfortunately the code to save a state dict for ZeRO is exorbitantly slow, so we also have to remove saving the optimizer state dicts.

## How multi node works

It actually works in much the same way as the single node multi GPU. Since in the single node setting we have multiple processes, now we are just adding extra processes on different machines.

The main differences here to consider are:
1. How to maintain the same environment on every node
2. How the nodes get in contact with each other (the `rdzv` arguments in the torchrun command)
3. How each node will access the data

Error reporting/handling becomes extremely important with more than 1 node. Networking issues are very common, and there are some subtle things that you need to ensure are identical between the machines.

## Shared storage - Managing your python virtual environment across nodes

For this the easiest approach is to create your python virtual environment in a shared network drive that all nodes can access. This way all of your nodes are using the exact same python executable/environment.

Creating the virtual environment is the same as normal, you just want the directory to be shared.

## Shared storage - Mangaging your dataset/model checkpoints across nodes

Again, the easiest approach here is to keep your data in a shared network drive. One thing to note is that shared network drives are slower to read from than node local drives. If you run into slowdowns in data loading, you can copy the data or model into node local storage.

When using `transformers` or `datasets`, make sure to set the `$HF_HOME` environment variable to control where huggingface downloads both datasets and model weights.

## `$HF_HOME` - The downloaded Model/Dataset directory

Huggingface `transformers` and `datasets` library will download things to `$HF_HOME` by default. `$HF_HOME` defaults to a **node local** value. There are two options for you here:

1. Keep `$HF_HOME` as node local and change `with rank0_first()` to be `with local_rank0_first()`
2. Change `$HF_HOME` to be a shared network drive

A third option which requires code changes to the code in this repo would be to do this automatically in code:

```python
@contextmanager
def rank_ordered(first: bool):
    if first:
        yield
    dist.barrier()
    if not first:
        yield
    dist.barrier()

# Determine if HF_HOME is node local or shared directory
hf_home_is_networked = os.path.ismount(os.environ["HF_HOME"])

if hf_home_is_networked:
    # We want rank 0 to go first (download will ONLY occur in rank 0) if directory is shared
    should_go_first = rank == 0
else:
    # If directory is node local we want a SINGLE process on the node to download the data (local rank 0)
    should_go_first = local_rank == 0

with rank_ordered(should_go_first):
    train_data = _load_and_preprocess_data(args, tokenizer, config)
```
