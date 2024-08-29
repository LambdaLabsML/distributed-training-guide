# Elastic Training

Elastic training is training where the launcher can restart a subset (or all) of the workers at various points throughout training.

Contrary to what you might think, usually when 1 worker encounters an error, **ALL workers are restarted** (see https://pytorch.org/docs/stable/elastic/run.html#membership-changes).

`torchrun` supports this via [elastic launch](https://pytorch.org/docs/stable/elastic/run.html#elastic-min-1-max-4-tolerates-up-to-3-membership-changes-or-failures):

```bash
torchrun
    --nnodes=1:4
    --max-restarts=3
    ...
```

which means that torchrun will restart all the workers up to 3 times (and if some of the nodes go offline, it can use as few as 1).

Note:
- `rank`, `local_rank`, and `world_size` are all not stable across restarts of a worker.
- Sometimes nodes have issues that can't be fixed just by restarting (like if you have a bug).

## Code Changes

No code changes are needed to do elastic training for our existing code. Instead it is more informative to play with a toy example where workers randomly crash to give you a sense for how it works.

```bash
cd distributed-training-guide/96-elastic-training
torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
    --max-restarts 3 \
    --redirects 3 \
    --log-dir ../logs \
    toy.py
```

This toy script will randomly throw an error from each of the ranks. **No GPU required to try this command!**

Inspect the log directory after you run this, for each attempt, there will be 1 worker sub directory that has a `error.json` file in it. You can also inspect each worker's stdout/stderr.
