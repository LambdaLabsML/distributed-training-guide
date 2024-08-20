# Diagnosing Errors

Hanging and deadlocks can be caused by so many things, even your own code! Here's some diagnostic tools that will help you figure out what is going on.

## Getting a dump of stack traces

Use [py-spy](https://github.com/benfred/py-spy) to get a dump of stacktraces from all python threads in a running python program. Here's how you get a dump from each worker:

```
py-spy dump --locals --pid <PID of the torchrun process>
```

Note that you have to run this on each node & for each worker you want a dump on.

## Recording errors

Python has a great built in library for getting errors that occur in any thread of a python program called [faulthandler](https://docs.python.org/3/library/faulthandler.html). This is especially useful when you're using a DataLoader with num_workers > 0.

Turns out, pytorch already has a built in way to use it! You just have to set `TORCHELASTIC_ERROR_FILE=./error.json` environment variable and add a `@record` annotation to your main function. 

```python
from torch.distributed.elastic.multiprocessing.errors import record

# NOTE: records errors to $TORCHELASTIC_ERROR_FILE
@record
def main():
    ...
```

Luckily all the code in this guide has been doing this, and so should you! **Make sure to set $TORCHELASTIC_ERROR_FILE**!.

## System metrics to watch for to diagnose hanging

`GPU Power Usage` will be the main one - if the training process is hanging, then the power usage will drop to around ~10% for all workers:

```
nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader
```

Will output something like this: (note this is with nothing running)
```
69.75 W, 700.00 W
75.10 W, 700.00 W
70.82 W, 700.00 W
69.29 W, 700.00 W
69.19 W, 700.00 W
68.72 W, 700.00 W
70.80 W, 700.00 W
70.87 W, 700.00 W
```

## Checklist for system problems

1. System date time on each system is the same (can cause NCCL timeouts)
2. NVLink valid topology `nvidia-smi topo -m`
3. NVLink status `nvidia-smi topo -p2p n` (additionally `w`/`r` in place of `n`)
4. Open file descriptor limit `ulimit -aH` (and then look for line containing `open files`).
5. `timeout` in `dist.init_process_group(timeout=...)` is sufficiently large.
