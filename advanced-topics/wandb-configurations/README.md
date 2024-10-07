# wandb configurations

There are a bunch of ways to configure wandb during your training runs. What will work best for you depends on how big your cluster is and what you want to track.

## rank 0

This is the standard approach. You will only see system information from the node that has the rank 0 process, and only data from rank 0 will be logged. It is minimal information, and you still get to track the experiment progress.

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

With this approach you can see system information from all nodes, and it scales linearly with number of nodes. This approach uses [wandb grouped runs](https://docs.wandb.ai/guides/runs/grouping/).

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

This configuration is really useful for tracking as much information about your cluster as possible. The downsides are that if you have a very large cluster, you can hit the ratelimit of wandb, and the wandb graphs become unusable.

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
