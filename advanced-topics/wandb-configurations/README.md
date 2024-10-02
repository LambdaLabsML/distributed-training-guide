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
