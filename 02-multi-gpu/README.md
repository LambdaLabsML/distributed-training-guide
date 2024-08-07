# Multi GPU on a single node

```bash
TORCHELASTIC_ERROR_FILE=./error.json OMP_NUM_THREADS=1 torchrun --standalone \
    --nnodes 1 \
    --nproc-per-node gpu \
    train_llm.py \
    --experiment-name gpt2-openwebtext-A100 \
    --dataset-name Skylion007/openwebtext \
    --model-name openai-community/gpt2 \
    --batch-size 64
```

## How Distributed Training works

Before we get into the changes required to do distributed training, let's think a little bit. If you've ever done parallel computations before, you know one way to achieve parallelization is to simply split your workload over all your cores. This is really useful if your task is relatively the same for all of the things you want to process. In fact, this is how python's multiprocessing.Pool.map object works.

Well distributed training with a GPU actually works the same way - we are splitting our workload (which is the batches from our dataset) over multiple GPUs.

However we have an additional problem: how do we ensure that the model on all of our GPUs is the same?

We can actually achieve this in a very clever way. For sake of simplicity let's assume:
1. Our model and optimizer fully fit on every GPU
2. We initialize our model the exact same way on all of our GPUs
3. Our optimizer has the exact same settings on all of our GPUs

Now let's focus on our training loop. The canonical one in pytorch is:

```python
loss = model(**batch)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The first 3 lines of the above can all be done asychronously. `loss.backward()` will compute gradients in each of our training processes. The clever bit is that `optimizer.step()` will synchronize the gradients across all processes before actually updating the model parameters.

So to be explicit: **`optimizer.step()` is a synchronization point across ALL processes**.

So how does pytorch achieve this?

### Splitting data across our workers - `torch.utils.data.distributed.DistributedSampler`

In our normal training script we use a `torch.utils.data.DataLoader` to batch our data. One of the arguments to DataLoader is a `sampler`, which basically samples items from the dataset when constructing the batches. You can think of the sampler as doing:

```
# simplified Sampler
worker_len = len(dataset)
random.choice(range(worker_len))
```

The clever thing that the DistributedSampler does is it partitions the length of the dataset across each of our workers. You don't even have to partition the actual dataset - it just chooses the integers that it returns from a specific subset of the dataset:

```
# simplified DistributedSampler
worker_len = len(dataset) // world_size
rank * worker_len + random.choice(range(worker_len))
```

### Gradient Synchronization - `torch.nn.parallel.DistributedDataParallel`

Funnily enough you might assume that the DDP module splits batches across processes, but that is not what it does at all!

This is a model wrapper class that sets up the gradient synchronization. I encourage you to read the documentation for this, it's very informative: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html.

This class also ensures that *model parameters are equal when you construct it*!

It achieves all of this through some very special model hooks.

## Code Changes

### Using `torchrun` instead of `python`

When training with multiple GPUs, we are actually spinning up a process for each GPU we are using. `torchrun` does this for us. It also manages the communication settings between each of the processes.

- `--standalone` argument is used when only running on a single node.
- `--nnodes` is the number of nodes we are using, in this case 1, but once we go to multiple nodes, this will be > 1.
- `--nproc-per-node` is the number of processes. `gpu` means to use all available GPUs.

#### OMP_NUM_THREADS

pytorch by default tries to take advantage of all the cores available when doing computations, even when you are on the GPU. Since we have multiple processes running pytorch, if we didn't set `OMP_NUM_THREADS` to 1, all of them would try to use all available cores.

Another way to do this is to call:

```python
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

in your training code, but torchrun will give a warning if you don't set the environment variable.

### wandb groups

While a lot of wandb usage in distributed training code seems to only initialize wandb from the main process, there's actually another way - groups! There are a ton of benefits from doing wandb in each process:
1. You can dive down into each worker's individual metrics/progress
2. Errors for each worker are logged.

## Code Diff

```diff
diff --git a/01-single-gpu/train_llm.py b/02-multi-gpu/train_llm.py
index 66240cc..d17fcb0 100644
--- a/01-single-gpu/train_llm.py
+++ b/02-multi-gpu/train_llm.py
@@ -10,6 +10,11 @@ import logging
 
 import torch
 from torch.utils.data import DataLoader
+from torch.utils.data.distributed import DistributedSampler
+from torch.nn.parallel import DistributedDataParallel
+from torch import distributed as dist
+from torch.distributed.elastic.multiprocessing.errors import record
+
 import numpy
 import wandb
 import tqdm
@@ -24,6 +29,7 @@ from transformers import (
 _LOGGER = logging.getLogger(__name__)
 
 
+@record
 def main():
     logging.basicConfig(level=logging.INFO)
 
@@ -32,13 +38,25 @@ def main():
 
     _LOGGER.info(args)
 
+    torch.set_num_threads(1)
+    torch.set_num_interop_threads(1)
+
     torch.manual_seed(args.seed)
     torch.cuda.manual_seed_all(args.seed)
     numpy.random.seed(args.seed)
     random.seed(args.seed)
 
-    device = torch.device("cuda")
+    dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "mpi")
+
+    rank = dist.get_rank()
+    world_size = dist.get_world_size()
+    assert world_size == torch.cuda.device_count()
+
+    _LOGGER.info(f"rank={rank} world size={world_size}")
+
+    device = torch.device(f"cuda:{rank}")
     dtype = torch.bfloat16
+    torch.cuda.set_device(device)
 
     def _load_to_device(p):
         return torch.load(p, map_location=device, weights_only=True)
@@ -53,19 +71,24 @@ def main():
     if len(tokenizer) > embedding_size:
         model.resize_token_embeddings(len(tokenizer))
 
-    train_data = _load_and_preprocess_data(args, tokenizer, config)
+    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
 
-    _LOGGER.info(f"{len(train_data)} training samples")
+    # NOTE: since this can download data, make sure to do the main process first
+    if rank == 0:
+        train_data = _load_and_preprocess_data(args, tokenizer, config)
+    dist.barrier()
+    if rank > 0:
+        train_data = _load_and_preprocess_data(args, tokenizer, config)
+    _LOGGER.info(f"[{rank}] {len(train_data)} training samples")
 
     dataloader = DataLoader(
         train_data,
         batch_size=args.batch_size,
-        shuffle=True,
-        drop_last=True,
         collate_fn=default_data_collator,
+        # NOTE: this sampler will split dataset evenly across workers
+        sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
     )
-
-    _LOGGER.info(f"{len(dataloader)} batches per epoch")
+    _LOGGER.info(f"[{rank}] {len(dataloader)} batches per epoch")
 
     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
@@ -73,7 +96,6 @@ def main():
     )
 
     exp_dir: Path = Path(args.save_dir) / args.experiment_name
-    _LOGGER.info(f"Experiment saving to {exp_dir}")
 
     # attempt resume
     state = {
@@ -90,15 +112,23 @@ def main():
         with open(exp_dir / "state.json") as fp:
             state = json.load(fp)
         resumed = True
+    _LOGGER.info(f"[{rank}] Resumed={resumed} | {state}")
+
+    dist.barrier()
+    if rank == 0:
+        _LOGGER.info(f"Creating experiment root directory")
+        exp_dir.mkdir(parents=True, exist_ok=True)
+    dist.barrier()
 
-    _LOGGER.info(f"Resumed={resumed} | {state}")
-    exp_dir.mkdir(parents=True, exist_ok=True)
+    (exp_dir / f"gpu-{rank}").mkdir(parents=True, exist_ok=True)
+    _LOGGER.info(f"[{rank}] Worker saving to {exp_dir / f'gpu-{rank}'}")
 
     wandb.init(
         project="distributed-training-tutorials",
-        dir=exp_dir,
-        name=args.experiment_name,
-        id=args.experiment_name,
+        dir=exp_dir / f"gpu-{rank}",
+        group=args.experiment_name,
+        name=args.experiment_name + "/" + f"gpu-{rank}",
+        id=f"{args.experiment_name}-{rank}",
         resume="must" if resumed else None,
         save_code=True,
         config={
@@ -106,15 +136,19 @@ def main():
             "embedding_size": len(tokenizer),
             "training_data_size": len(train_data),
             "num_batches": len(dataloader),
+            "rank": rank,
+            "world_size": world_size,
         },
     )
 
     timers = {k: LocalTimer(device) for k in ["data", "forward", "backward", "update"]}
 
     for state["epoch"] in range(state["epoch"], args.num_epochs):
-        _LOGGER.info(f"Begin epoch {state['epoch']} at step {state['epoch_step']}")
+        _LOGGER.info(
+            f"[{rank}] Begin epoch {state['epoch']} at step {state['epoch_step']}"
+        )
 
-        progress_bar = tqdm.tqdm(range(len(dataloader)))
+        progress_bar = tqdm.tqdm(range(len(dataloader)), disable=rank > 0)
         if state["epoch_step"] > 0:
             progress_bar.update(state["epoch_step"])
         for i_step, batch in enumerate(dataloader):
@@ -162,11 +196,13 @@ def main():
                     t.reset()
 
             if state["global_step"] % args.ckpt_freq == 0:
-                torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
-                torch.save(model.state_dict(), exp_dir / "model.pt")
-                torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
-                with open(exp_dir / "state.json", "w") as fp:
-                    json.dump(state, fp)
+                if rank == 0:
+                    torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
+                    torch.save(model.state_dict(), exp_dir / "model.pt")
+                    torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
+                    with open(exp_dir / "state.json", "w") as fp:
+                        json.dump(state, fp)
+                dist.barrier()
 
         state["epoch_step"] = 0
 
@@ -268,4 +304,7 @@ def _get_parser() -> argparse.ArgumentParser:
 
 
 if __name__ == "__main__":
-    main()
+    try:
+        main()
+    finally:
+        dist.destroy_process_group()
```