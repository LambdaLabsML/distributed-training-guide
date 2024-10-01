# Job Launchers

Since it is quite cumbersome to manually SSH into every node and start a training job, there are various ways to launch distributed training jobs from a single node.

NOTE: This chapter's code builds off of chapter 3's code.

## deepspeed

deepspeed is a distributed training library with many optimizations. We go into some of these optimizations in more detail in later chapters, but here we can just use the launcher included with it.

**NOTE: you do not have to integrate deepspeed into your training code to use the deepspeed launcher.**

1. Install: `pip install deepspeed`
2. Add `--local_rank` to cli parsing:

```diff
     parser.add_argument("--log-freq", default=100, type=int)
     parser.add_argument("--ckpt-freq", default=500, type=int)
+    parser.add_argument("--local_rank", type=int, default=None)
     return parser
```

3. Use it when initializing local_rank

```diff
-    local_rank = rank % torch.cuda.device_count()
+    local_rank = args.local_rank or (rank % torch.cuda.device_count())
```

4. Launch

```bash
cd distributed-training-guide/04-job-launchers-deepspeed
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
