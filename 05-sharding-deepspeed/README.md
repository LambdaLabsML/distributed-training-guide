# Sharding Across GPUs

Up to this point we have assumed that both the model & optimizer fully fit on a single GPU. So each GPU during our training process fully contains a copy of the model and optimizer.

This becomes an issue once the model becomes big enough - either the model itself cannot fit, or the optimizer (which usually contains 1-4x the memory of the model) cannot fit anymore.

Sharding refers to spreading the **storage** of a combination of: optimizer state, gradients, and/or model parameters **across your GPUs**. **The execution of layers DOES NOT CHANGE**.

What this means:

1. Each layer of your model still needs to pull the **entire** layer's parameters/gradients/optimizer states into GPU memory. After the layer is done, then those pieces are resharded.
2. There are synchronization costs to un-shard and re-shard before and after each layer.
3. Sharding does not reduce the peak memory cost of your biggest layer.

**Sharding is a data parallel technique! NOT a model/tensor/pipeline parallel technique**

## DeepSpeed ZeRO

![image](<img width="583" alt="image" src="https://github.com/user-attachments/assets/c2966a46-1807-4a56-92d4-977798087dd6">)

This is actually a collection of modes to shard more and more memory:

> ZeRO Stage 1: The optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition.

> ZeRO Stage 2: The reduced 32-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states.

> ZeRO Stage 3: The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes.

References:
- [deepspeed docs](https://deepspeed.readthedocs.io/en/latest/zero3.html)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)

### Integrating DeepSpeed into training code

#### Argument Parsing

```diff
@@ -305,11 +302,10 @@ def _get_parser() -> argparse.ArgumentParser:
     parser.add_argument("--log-freq", default=100, type=int)
     parser.add_argument("--ckpt-freq", default=500, type=int)
     parser.add_argument("--dataset-cache-root", default="../.cache")
+    parser.add_argument("--local_rank", type=int, default=None)
+    deepspeed.add_config_arguments(parser)
     return parser
```

#### Initialization

Two main differences here:
1. We call `deepspeed.init_distributed` instead of using pytorch's `init_process_group`
2. We call `deepspeed.initialize` after we've constructed the model **instead** of wrapping the model with DDP.

**NOTE**: `deepspeed.initialize` will construct the optimizer & lr_scheduler based on the config you pass in

```diff
@@ -14,6 +14,7 @@ from torch.nn.parallel import DistributedDataParallel
 from torch import distributed as dist
 from torch.distributed.elastic.multiprocessing.errors import record
 
+import deepspeed
 import numpy
 import wandb
 import tqdm
@@ -42,10 +43,15 @@ def main():
     numpy.random.seed(args.seed)
     random.seed(args.seed)
 
-    dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "mpi")
+    deepspeed.init_distributed(
+        dist_backend="nccl" if dist.is_nccl_available() else "mpi"
+    )
 
     rank = dist.get_rank()
-    local_rank = rank % torch.cuda.device_count()
+    if args.local_rank is not None:
+        local_rank = args.local_rank
+    else:
+        local_rank = rank % torch.cuda.device_count()
     world_size = dist.get_world_size()
 
     _LOGGER.info(f"local rank={local_rank} rank={rank} world size={world_size}")
 
@@ -73,10 +73,6 @@ def main():
     if len(tokenizer) > embedding_size:
         model.resize_token_embeddings(len(tokenizer))
 
-    model = DistributedDataParallel(
-        model, device_ids=[local_rank], output_device=local_rank
-    )
-
@@ -89,9 +95,11 @@ def main():
     )
     _LOGGER.info(f"[{rank}] {len(dataloader)} batches per epoch")
 
-    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
-    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
-        optimizer, T_max=1000, eta_min=args.lr * 1e-2
+    model_engine: deepspeed.DeepSpeedEngine
+    model_engine, _, _, lr_scheduler = deepspeed.initialize(
+        args,
+        model=model,
+        model_parameters=(p for p in model.parameters() if p.requires_grad),
     )
```

#### Train Loop

Here we are just going to be replacing our pytorch calls with deepspeed calls. Note that we don't have direct access to optimizer/lr_scheduler anymore since deepspeed handles that.

```diff
             with timers["forward"]:
-                outputs = model(**batch)
+                outputs = model_engine(**batch)
 
             with timers["backward"]:
-                optimizer.zero_grad()
-                outputs.loss.backward()
+                model_engine.backward(outputs.loss)
 
             with timers["update"]:
-                optimizer.step()
-                lr_scheduler.step()
+                model_engine.step()
 
             state["global_step"] += 1
             state["epoch_step"] += 1
```

#### Checkpoints

Loading becomes:

```diff
     resumed = False
-    if (exp_dir / "state.json").exists():
-        model.load_state_dict(_load_to_device(exp_dir / "model.pt"))
-        optimizer.load_state_dict(_load_to_device(exp_dir / "optimizer.pt"))
-        lr_scheduler.load_state_dict(_load_to_device(exp_dir / "lr_scheduler.pt"))
-        with open(exp_dir / "state.json") as fp:
-            state = json.load(fp)
-        resumed = True
+    if (exp_dir / "pytorch_model.bin").exists():
+        load_path, state = model_engine.load_checkpoint(exp_dir)
+        resumed = load_path is not None
```

Saving becomes: (**NOTE**: saving must be done on ALL ranks instead of just rank 0 - because of sharding)

```diff
             if state["global_step"] % args.ckpt_freq == 0:
-                if rank == 0:
-                    torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
-                    torch.save(model.state_dict(), exp_dir / "model.pt")
-                    torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
-                    with open(exp_dir / "state.json", "w") as fp:
-                        json.dump(state, fp)
+                model_engine.save_checkpoint(exp_dir, client_state=state)
                 dist.barrier()
```

### Configuration

```json
{
    "train_micro_batch_size_per_gpu": 64,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 3e-5
        }
    },
    "scheduler": {
        "type": "WarmupCosineLR",
        "params": {
            "total_num_steps": 1000,
            "warmup_num_steps": 0,
            "cos_min_ratio": 1e-2
        }
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3
    }
}
```

### Command

```bash
cd distributed-training-guide/05-sharding-deepspeed
TORCHELASTIC_ERROR_FILE=../error.json OMP_NUM_THREADS=1 deepspeed \
    --enable_each_rank_log ../logs \
    train_llm.py \
    --experiment-name deepspeed-multi-node-$(date +%Y-%m-%dT%H-%M-%S) \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2 \
    --deepspeed_config ds_config.json
```
