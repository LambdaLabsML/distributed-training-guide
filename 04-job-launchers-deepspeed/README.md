# Job Launchers

Since it is quite cumbersome to manually SSH into every node and start a training job, there are various ways to launch distributed training jobs from a single node.

## deepspeed

deepspeed is a distributed training library with many optimizations. We go into some of these optimizations in more detail in later chapters, but here we can just use the launcher included with it.

**NOTE: you do not have to integrate deepspeed into your training code to use the deepspeed launcher.**

1. Install: `pip install deepspeed`
2. Add `--local_rank` to cli parsing:

```diff --git a/../03-multi-node/train_llm.py b/train_llm.py
index ae1c66f..d5671b3 100644
--- a/../03-multi-node/train_llm.py
+++ b/train_llm.py
@@ -49,7 +49,10 @@ def main():
     dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "mpi")
 
     rank = dist.get_rank()
-    local_rank = rank % torch.cuda.device_count()
+    if args.local_rank is not None:
+        local_rank = args.local_rank
+    else:
+        local_rank = rank % torch.cuda.device_count()
     world_size = dist.get_world_size()
 
     _LOGGER.info(f"local_rank={local_rank} rank={rank} world size={world_size}")
@@ -306,6 +309,7 @@ def _get_parser() -> argparse.ArgumentParser:
     parser.add_argument("--log-freq", default=100, type=int)
     parser.add_argument("--ckpt-freq", default=500, type=int)
     parser.add_argument("--dataset-cache-root", default="../.cache")
+    parser.add_argument("--local_rank", type=int, default=None)
     return parser
```

4. Launch

```bash
cd distributed-training-guide/04-job-launchers-deepspeed
TORCHELASTIC_ERROR_FILE=../error.json OMP_NUM_THREADS=1 deepspeed \
    --include <ip of node 1>@<ip of node 2> \
    --enable_each_rank_log ../logs \
    train_llm.py \
    --experiment-name deepspeed-multi-node \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2 \
    --batch-size 64
```
