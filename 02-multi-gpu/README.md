# Multi GPU on a single node

**NOTE: This chapter's code builds off of [chapter 1](../01-single-gpu).**

```bash
cd distributed-training-guide/02-multi-gpu
export TORCHELASTIC_ERROR_FILE=../error.json
export OMP_NUM_THREADS=1
torchrun --standalone \
    --nnodes 1 \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ../logs \
    train_llm.py \
    --experiment-name gpt2-alpaca-multi-gpu-$(date +%Y-%m-%dT%H-%M-%S) \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2
```

## Dictionary

- `world size`: the total number of participating gpus
- `rank` the global unique id of this worker (from `0` up to and including `world_size - 1`)
- `local_rank` the rank local to this machine (from `0` up to and including `torch.cuda.device_count() - 1`)

## How Distributed Training works

Before we get into the changes required to do distributed training, let's think a little bit. If you've ever done parallel computations before, you know one way to achieve parallelization is to simply split your workload over all your cores. This is really useful if your task is relatively the same for all of the things you want to process. In fact, this is how python's multiprocessing.Pool.map object works.

Well distributed training with a GPU actually works the same way - we are splitting our workload (which is the batches from our dataset) over multiple GPUs.

<img width="818" alt="image" src="https://github.com/user-attachments/assets/acaaef83-28f4-4d1d-bbeb-0a6dc5a555bf">

However we have an additional problem: how do we ensure that the model on all of our GPUs is the same?

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

So to be explicit: **`optimizer.step()` is a synchronization point across ALL processes**.

So how does pytorch achieve this?

### Running N copies of our training script

When we use `torchrun` to launch a distributed training job, what's happening is that it is **launch N separate processes** (where N is the number of gpus you specify in `--nproc-per-node`), all running your same training script.

<img width="611" alt="image" src="https://github.com/user-attachments/assets/44919819-144c-4498-b33f-78dfde388c31">

It will also set up some synchronization between the processes (the `--standlone` argument for now, but we will learn more in chapter 3).

Then each of the processes is running the same training code and needs to synchronize at various points.

Each of the processes gets an id (the `rank`), which will tell it what device to use.

### Splitting data across our workers - `torch.utils.data.distributed.DistributedSampler`

In our normal training script we use a `torch.utils.data.DataLoader` to batch our data. One of the arguments to DataLoader is a `sampler`, which basically samples items from the dataset when constructing the batches. You can think of the sampler as doing:

```python
# simplified Sampler
worker_len = len(dataset)
random.choice(range(worker_len))
```

The clever thing that the DistributedSampler does is it partitions the length of the dataset across each of our workers. You don't even have to partition the actual dataset - it just chooses the integers that it returns from a specific subset of the dataset:

```python
# simplified DistributedSampler
worker_len = len(dataset) // dist.get_world_size()
dist.get_rank() * worker_len + random.choice(range(worker_len))
```

### Gradient Synchronization - `torch.nn.parallel.DistributedDataParallel`

Funnily enough you might assume that the DDP module splits batches across processes, but that is not what it does at all!

This is a model wrapper class that ensures **gradients are synchronized before calling optimizer.step()**. I encourage you to read the documentation for this, it's very informative: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html.

This class also ensures that *model parameters are equal when you construct it*!

It achieves all of this through some [very special model hooks](https://github.com/pytorch/pytorch/blob/v2.4.1/torch/nn/parallel/distributed.py#L939) to sum all the gradients from all the ranks on all the ranks together:

```python
# NOTE: internal pytorch code found at https://github.com/pytorch/pytorch/blob/v2.4.1/torch/nn/parallel/distributed.py#L939
gradient = param.grad / self.process_group.size()
gradient = fcol.all_reduce(gradient, "sum", self.process_group)
```

## Code Changes

### Using `torchrun` instead of `python`

When training with multiple GPUs, we are actually spinning up a process for each GPU we are using. `torchrun` does this for us. It also manages the communication settings between each of the processes.

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

### Calling `dist.init_process_group()`

One of the main changes is including `dist.init_process_group()`. You are required to call this before calling other dist apis.

```diff
 def main():
     parser = _get_parser()
     args = parser.parse_args()
+    dist.init_process_group()
```

Note that we are now:
1. Setting our device using `rank`: `device = torch.device(f"cuda:{rank}")`
2. Calling `torch.cuda.set_device(device)`, which is **required for dist calls to work**.

```diff
-device = torch.device(f"cuda")
+device = torch.device(f"cuda:{rank}")
 dtype = torch.bfloat16
+torch.cuda.set_device(device)
```

### Using DistributedDataParallel

As discussed earlier - this is for gradient synchronization and model weight syncing at initialization. We just call this after we've already constructed our models.

```diff
 model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype).to(device)
+model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
```

### Downloading data in rank 0 first

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
-config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
-model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype).to(device)
-tokenizer = AutoTokenizer.from_pretrained(args.model_name)
+with rank0_first():
+    config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
+    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype).to(device)
+    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
```

Downloading data:

```diff
-train_data = _load_and_preprocess_data(args, tokenizer, config)
+# NOTE: since this can download data, make sure to do the main process first
+with rank0_first():
+    train_data = _load_and_preprocess_data(args, tokenizer, config)
```

### Using DistributedSampler

As discussed before, this will let each rank grab a different subset of the data.

```diff
 dataloader = DataLoader(
     train_data,
     batch_size=args.batch_size,
-    shuffle=True,
-    drop_last=True,
     collate_fn=default_data_collator,
+    sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
 )
```

### Only creating experiment directory on rank 0

Note the `dist.barrier()` calls before and after we create the directory. **These are very important!**

Since we check to see if the experiment directory already exists right before creating the experiment directory, we need to ensure that **all processes have checked for its existence**. So the first `dist.barrier()` call ensures that all workers have already checked the existence of that. Then and only then can we create the directory on rank 0.

```diff
-exp_dir.mkdir(parents=True, exist_ok=True)
+if rank == 0:
+    exp_dir.mkdir(parents=True, exist_ok=True)
+dist.barrier()
```

### Grouped wandb runs

wandb allows you to create groups of runs, all grouped under a single unique group name. Each one of our workers will be calling `wandb.init()` with the same group name. Then we can upload information from each worker to wandb, and visualize them all together!

Another standard method is to only call wandb.init and wandb.log on rank 0, but it is helpful for debugging to see the stats from each of the worker processes.

See our chapter on [wandb-configurations](../advanced-topics/wandb-configurations/) for more details.

```diff
wandb.init(
         project="distributed-training-guide",
+        group=args.experiment_name,
-        dir=exp_dir,
+        dir=exp_dir / f"rank-{rank}",
-        name=args.experiment_name,
+        name=f"rank-{rank}",
-        id=args.experiment_name,
+        id=f"{args.experiment_name}-{rank}",
         resume="must" if resumed else None,
         save_code=True,
         config={
             "args": vars(args),
             "training_data_size": len(train_data),
             "num_batches": len(dataloader),
+            "rank": rank,
+            "world_size": world_size,
         },
     )
```

### Save checkpoint on rank 0

We only want one of our ranks to save a checkpoint. Otherwise the ranks might write to the same file and corrupt each other.

```diff
 if state["global_step"] % args.ckpt_freq == 0:
-    torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
-    torch.save(model.state_dict(), exp_dir / "model.pt")
-    torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
-    with open(exp_dir / "state.json", "w") as fp:
-        json.dump(state, fp)
+    if rank == 0:
+        torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
+        torch.save(model.state_dict(), exp_dir / "model.pt")
+        torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
+        with open(exp_dir / "state.json", "w") as fp:
+             json.dump(state, fp)
+    dist.barrier()
```
