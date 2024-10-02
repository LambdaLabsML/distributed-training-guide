# Diagnosing Errors

Hanging and deadlocks can be caused by so many things, even your own code! Here's some diagnostic tools that will help you figure out what is going on.

## System metrics to watch for to diagnose hanging

`GPU Power Usage` will be the main one - if the training process is hanging, then the power usage will drop to around ~10% for all workers:

```bash
> nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader
69.75 W, 700.00 W
75.10 W, 700.00 W
70.82 W, 700.00 W
69.29 W, 700.00 W
69.19 W, 700.00 W
68.72 W, 700.00 W
70.80 W, 700.00 W
70.87 W, 700.00 W
```

Using our provided [top-cluster.py](../top-cluster.py) script will output something like this:

```bash
> python top-cluster.py <hosts file>
===2024-10-02 19:55:02.553039
    name	      util	     power	    memory	    nprocs
 cluster	    100.0%	     99.1%	     96.9%	        64
node-001	    100.0%	     99.7%	     96.1%	         8
node-002	    100.0%	     97.8%	     96.9%	         8
node-003	    100.0%	     99.2%	     97.2%	         8
node-004	    100.0%	     99.1%	     97.4%	         8
node-005	    100.0%	     98.1%	     97.1%	         8
node-006	    100.0%	     99.0%	     97.7%	         8
node-007	    100.0%	     99.8%	     96.9%	         8
node-008	    100.0%	    100.0%	     96.2%	         8
===
```

## Getting a dump of stack traces

Use [py-spy](https://github.com/benfred/py-spy) to get a dump of stacktraces from all python threads in a running python program. Here's how you get a dump from each worker:

```
sudo env "PATH=$PATH" py-spy dump --locals --pid <PID of the torchrun process>
```

## Benchmarking/profiling

You can use `py-spy top --pid <>`, to get a `top`/`htop` like view of the functions that are being called in your python process.

## Recording errors

Python has a great built in library for getting errors that occur in any thread of a python program called [faulthandler](https://docs.python.org/3/library/faulthandler.html). This is especially useful when you're using a DataLoader with num_workers > 0.

Turns out, pytorch already has a built in way to use it! You just have to set `TORCHELASTIC_ERROR_FILE=../error.json` environment variable and add a `@record` annotation to your main function. 

```python
from torch.distributed.elastic.multiprocessing.errors import record

# NOTE: records errors to $TORCHELASTIC_ERROR_FILE
@record
def main():
    ...
```

Luckily all the code in this guide has been doing this, and so should you! **Make sure to set $TORCHELASTIC_ERROR_FILE**!.

## Checklist for system problems

1. System date time on each system is the same (can cause NCCL timeouts)
2. NVLink valid topology `nvidia-smi topo -m`
3. NVLink status `nvidia-smi topo -p2p n` (additionally `w`/`r` in place of `n`)
4. Open file descriptor limit `ulimit -aH` (and then look for line containing `open files`).
5. `timeout` in `dist.init_process_group(timeout=...)` is sufficiently large.
