# Multi GPU across multiple nodes

Run this command on **every** participating node

```bash
TORCHELASTIC_ERROR_FILE=./error.json OMP_NUM_THREADS=1 torchrun \
    --rdzv-id multi-node \
    --rdzv-backend c10d \
    --rdzv-endpoint <IP ADDRESS of main node>:<port> \
    --nnodes 2 \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ./logs \
    train_llm.py \
    --experiment-name multi-node \
    --dataset-name Skylion007/openwebtext \
    --model-name openai-community/gpt2 \
    --batch-size 64
```

Assumes:
1. You are using the same enviroment on both machines
2. T
3. You are logged into wandb on both machines


## How Multi Node works

It actually works in much the same way as the multi GPU. Since in the single node setting we have multiple processes, now we are just adding extra processes on different machines.

The main differences here to consider are:
1. How the nodes get in contact with each other (the `rdzv` arguments in the torchrun command)
2. Your code may need to use `local_rank` instead of `rank`. `rank` is between 0 and world_size, so if you have 2 machines, the second machine may have ranks 8-16. Local rank on the second machine will still be 0-8.

Error reporting/handling becomes extremely important with more than 1 node. Networking issues are very common, and there are some subtle things that you need to ensure are identical between the machines.

tl;dr: When going from single to multi node, ensuring environments are the same is the most important thing.

## Machine Setup

### Ensuring Environments are the same

### Ensuring System Dates & Times are the same

## Troubleshooting

How to set up your code/logging to make it easy to identify the cause?

### Networking Issues

### GPU Issues

### Hanging/Timeout issues

## Code Diff

```diff
diff --git a/02-multi-gpu/train_llm.py b/03-multi-node/train_llm.py
index d17fcb0..e593b16 100644
--- a/02-multi-gpu/train_llm.py
+++ b/03-multi-node/train_llm.py
@@ -49,12 +49,12 @@ def main():
     dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "mpi")
 
     rank = dist.get_rank()
+    local_rank = rank % torch.cuda.device_count()
     world_size = dist.get_world_size()
-    assert world_size == torch.cuda.device_count()
 
-    _LOGGER.info(f"rank={rank} world size={world_size}")
+    _LOGGER.info(f"local rank={local_rank} rank={rank} world size={world_size}")
 
-    device = torch.device(f"cuda:{rank}")
+    device = torch.device(f"cuda:{local_rank}")
     dtype = torch.bfloat16
     torch.cuda.set_device(device)
 
@@ -71,9 +71,12 @@ def main():
     if len(tokenizer) > embedding_size:
         model.resize_token_embeddings(len(tokenizer))
 
-    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
+    model = DistributedDataParallel(
+        model, device_ids=[local_rank], output_device=local_rank
+    )
 
     # NOTE: since this can download data, make sure to do the main process first
+    # NOTE: This assumes that the data is on a **shared** network drive, accessible to all processes
     if rank == 0:
         train_data = _load_and_preprocess_data(args, tokenizer, config)
     dist.barrier()
@@ -85,6 +88,7 @@ def main():
         train_data,
         batch_size=args.batch_size,
         collate_fn=default_data_collator,
+        num_workers=1,
         # NOTE: this sampler will split dataset evenly across workers
         sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
     )
@@ -104,6 +108,7 @@ def main():
         "epoch_step": 0,
         "running_loss": 0,
     }
+
     resumed = False
     if (exp_dir / "model.pt").exists():
         model.load_state_dict(_load_to_device(exp_dir / "model.pt"))
@@ -116,18 +121,18 @@ def main():
 
     dist.barrier()
     if rank == 0:
-        _LOGGER.info(f"Creating experiment root directory")
+        # NOTE: assuming directory is shared across all nodes, that's why we do rank instead of local_rank
         exp_dir.mkdir(parents=True, exist_ok=True)
     dist.barrier()
 
-    (exp_dir / f"gpu-{rank}").mkdir(parents=True, exist_ok=True)
-    _LOGGER.info(f"[{rank}] Worker saving to {exp_dir / f'gpu-{rank}'}")
+    (exp_dir / f"rank-{rank}").mkdir(parents=True, exist_ok=True)
+    _LOGGER.info(f"[{rank}] Worker saving to {exp_dir / f'rank-{rank}'}")
 
     wandb.init(
         project="distributed-training-tutorials",
-        dir=exp_dir / f"gpu-{rank}",
+        dir=exp_dir / f"rank-{rank}",
         group=args.experiment_name,
-        name=args.experiment_name + "/" + f"gpu-{rank}",
+        name=f"rank-{rank}",
         id=f"{args.experiment_name}-{rank}",
         resume="must" if resumed else None,
         save_code=True,
@@ -137,6 +142,7 @@ def main():
             "training_data_size": len(train_data),
             "num_batches": len(dataloader),
             "rank": rank,
+            "local_rank": local_rank,
             "world_size": world_size,
         },
     )
```