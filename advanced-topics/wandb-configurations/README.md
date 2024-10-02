# wandb configurations

**NOTE: This chapter's code builds off of [chapter 3](../../03-multi-node/)'s code.**

There are a bunch of ways to configure wandb during your training runs. What will work best for you depends on how big your cluster is.

Recommendation is:

1. For large clusters (> 32 GPUs), use local_rank 0 if you'd like to track system metrics across the cluster with wandb, or use rank 0 if you want to minimize wandb usage
2. For everything else, use [grouped runs](https://docs.wandb.ai/guides/runs/grouping)

## rank 0

```python
if rank == 0:
    wandb.init(
        project="distributed-training-guide",
        dir=exp_dir,
        id=args.experiment_name,
        name=args.experiment_name,
        resume="must" if resumed else None,
        save_code=True,
        config=...,
    )
```

## local_rank 0 (every node)

```python
if local_rank == 0:
    wandb.init(
        project="distributed-training-guide",
        dir=exp_dir / f"rank-{rank}",
        group=args.experiment_name,
        name=f"rank-{rank}",
        id=f"{args.experiment_name}-{rank}",
        resume="must" if resumed else None,
        save_code=True,
        config=...,
    )
```

If you want the name to appear as the node id you can set:

```python
name=f"node-{rank // world_size}"
```

## every rank

[Grouping docs](https://docs.wandb.ai/guides/runs/grouping)

This is the configuration that the whole guide uses. It's really useful for tracking as much informatino about your cluster as possible. The downsides are that if you have a very large cluster, you can hit the ratelimit of wandb, and the wandb graphs become unusable.

```python
wandb.init(
    project="distributed-training-guide",
    dir=exp_dir / f"rank-{rank}",
    group=args.experiment_name,
    name=f"rank-{rank}",
    id=f"{args.experiment_name}-{rank}",
    resume="must" if resumed else None,
    save_code=True,
    config=...,
)
```

## Code changes to make this configurable

```diff --git a/03-multi-node/train_llm.py b/96-wandb-configurations/train_llm.py
index 38f3cf0..3233f81 100644
--- a/03-multi-node/train_llm.py
+++ b/96-wandb-configurations/train_llm.py
@@ -123,24 +123,32 @@ def main():
     (exp_dir / f"rank-{rank}").mkdir(parents=True, exist_ok=True)
     LOGGER.info(f"Worker saving to {exp_dir / f'rank-{rank}'}")
 
-    wandb.init(
-        project="distributed-training-guide",
-        dir=exp_dir / f"rank-{rank}",
-        group=args.experiment_name,
-        name=f"rank-{rank}",
-        id=f"{args.experiment_name}-{rank}",
-        resume="must" if resumed else None,
-        save_code=True,
-        config={
-            "args": vars(args),
-            "embedding_size": len(tokenizer),
-            "training_data_size": len(train_data),
-            "num_batches": len(dataloader),
-            "rank": rank,
-            "local_rank": local_rank,
-            "world_size": world_size,
-        },
-    )
+    if args.wandb_mode == "all-ranks":
+        use_wandb = True
+    elif args.wandb_mode == "every-node":
+        use_wandb = local_rank == 0
+    elif args.wandb_mode == "rank-0":
+        use_wandb = rank == 0
+
+    if use_wandb:
+        wandb.init(
+            project="distributed-training-guide",
+            dir=exp_dir / f"rank-{rank}",
+            group=args.experiment_name,
+            name=f"rank-{rank}",
+            id=f"{args.experiment_name}-{rank}",
+            resume="must" if resumed else None,
+            save_code=True,
+            config={
+                "args": vars(args),
+                "embedding_size": len(tokenizer),
+                "training_data_size": len(train_data),
+                "num_batches": len(dataloader),
+                "rank": rank,
+                "local_rank": local_rank,
+                "world_size": world_size,
+            },
+        )
 
     timers = {k: LocalTimer(device) for k in ["data", "forward", "backward", "update"]}
 
@@ -181,21 +189,24 @@ def main():
             progress_bar.update(1)
 
             if state["global_step"] % args.log_freq == 0:
-                wandb.log(
-                    {
-                        "lr": lr_scheduler.get_last_lr()[0],
-                        "running_loss": state["running_loss"] / args.log_freq,
-                        "epoch": state["epoch"],
-                        "epoch_progress": state["epoch_step"] / len(dataloader),
-                        "num_batches_remaining": len(dataloader) - i_step,
-                        "time/total": sum(t.avg_elapsed_ms() for t in timers.values()),
-                        **{
-                            f"time/{k}": timer.avg_elapsed_ms()
-                            for k, timer in timers.items()
-                        },
+                info = {
+                    "lr": lr_scheduler.get_last_lr()[0],
+                    "running_loss": state["running_loss"] / args.log_freq,
+                    "epoch": state["epoch"],
+                    "epoch_progress": state["epoch_step"] / len(dataloader),
+                    "num_batches_remaining": len(dataloader) - i_step,
+                    "time/total": sum(t.avg_elapsed_ms() for t in timers.values()),
+                    **{
+                        f"time/{k}": timer.avg_elapsed_ms()
+                        for k, timer in timers.items()
                     },
-                    step=state["global_step"],
-                )
+                }
+
+                if use_wandb:
+                    wandb.log(info, step=state["global_step"])
+                else:
+                    print(f"step={state['global_step']} | {info}")
+
                 state["running_loss"] = 0
                 for t in timers.values():
                     t.reset()
@@ -305,6 +316,11 @@ def _get_parser() -> argparse.ArgumentParser:
     parser.add_argument("--log-freq", default=100, type=int)
     parser.add_argument("--ckpt-freq", default=500, type=int)
+    parser.add_argument(
+        "--wandb-mode",
+        default="all-ranks",
+        choices=["all-ranks", "every-node", "rank-0"],
+    )
     return parser
```
