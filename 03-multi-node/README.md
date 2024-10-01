# Multi GPU across multiple nodes

NOTE: This chapter's code builds off of chapter 2.

Run this command on **every** participating node.

```bash
cd distributed-training-guide/03-multi-node
export HF_HOME=../.cache
export TORCHELASTIC_ERROR_FILE=../error.json
export OMP_NUM_THREADS=1
torchrun \
    --rdzv-id multi-node \
    --rdzv-backend c10d \
    --rdzv-endpoint <IP ADDRESS of main node>:<port> \
    --nnodes 2 \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ../logs \
    train_llm.py \
    --experiment-name gpt2-alpaca-multi-node-$(date +%Y-%m-%dT%H-%M-%S) \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2
```

Assumes:
1. You are using the same enviroment on both machines
2. You are logged into wandb on both machines

## How Multi Node works

It actually works in much the same way as the multi GPU. Since in the single node setting we have multiple processes, now we are just adding extra processes on different machines.

The main differences here to consider are:
1. How the nodes get in contact with each other (the `rdzv` arguments in the torchrun command)
2. Your code may need to use `local_rank` instead of `rank`. `rank` is between 0 and world_size, so if you have 2 machines, the second machine may have ranks 8-16. Local rank on the second machine will still be 0-8.
3. How each node has a copy of the same data (either downloading the data to a shared network drive, or downloading a copy to each node)

Error reporting/handling becomes extremely important with more than 1 node. Networking issues are very common, and there are some subtle things that you need to ensure are identical between the machines.

tl;dr: When going from single to multi node, ensuring environments are the same is the most important thing.

### Shared network drive

Shared network drives are the easiest way to maintain the same data/environment on all nodes. When you create a python virtual environment in a shared network drive, all nodes will be able to use the same python executable.

Additionally, you can put all of your data and code in the shared directory as well.

Whatever you do, make sure to set the `HF_HOME` environment variable to control where huggingface downloads both datasets and model weights.

## Code Diff

Not much has to change code wise. The main thing is how your data is stored/organized.

If your workers operate on shared network space, then only `rank==0` should be writing to it. Otherwise you may want `local_rank==0` writing, so each node is collecting results.

```diff --git a/02-multi-gpu/train_llm.py b/03-multi-node/train_llm.py
index 3130381..d5cb05c 100644
--- a/02-multi-gpu/train_llm.py
+++ b/03-multi-node/train_llm.py
@@ -49,12 +49,12 @@ def main():
     dist.init_process_group()
 
     rank = dist.get_rank()
+    local_rank = rank % torch.cuda.device_count()
     world_size = dist.get_world_size()
-    assert world_size == torch.cuda.device_count()
 
-    _LOGGER.info(f"rank={rank} world size={world_size}")
+    _LOGGER.info(f"local_rank={local_rank} rank={rank} world size={world_size}")
 
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
@@ -116,6 +119,7 @@ def main():
 
     dist.barrier()
     if rank == 0:
+        # NOTE: assuming directory is shared across all nodes, that's why we do rank instead of local_rank
         _LOGGER.info(f"Creating experiment root directory")
         exp_dir.mkdir(parents=True, exist_ok=True)
     dist.barrier()
@@ -137,6 +141,7 @@ def main():
             "training_data_size": len(train_data),
             "num_batches": len(dataloader),
             "rank": rank,
+            "local_rank": local_rank,
             "world_size": world_size,
         },
     )
```