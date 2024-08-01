# Multi GPU on a single node

```bash
OMP_NUM_THREADS=1 torchrun --standalone \
    --nnodes 1 \
    --nproc-per-node gpu \
    train_llm.py \
    --experiment-name gpt2-openwebtext-A100 \
    --dataset-name Skylion007/openwebtext \
    --model-name openai-community/gpt2 \
    --batch-size 64
```

## How Distributed Training works

Before we get into the changes required to do distributed training, let's think a little bit. If you've ever done parallel computations before, you know one way to achieve parallelization is to simply split your workload over all your cores. This is really useful if your task is relatively the same for all of the things you want to process. In fact, this is how python's multiprocessing.Pool.map object works.

Well distributed training with a GPU actually works the same way - we are splitting our workload (which is the batches from our dataset) over multiple GPUs.

However we have an additional problem: how do we ensure that the model on all of our GPUs is the same?

We can actually achieve this in a very clever way. For sake of simplicity let's assume:
1. Our model and optimizer fully fit on every GPU
2. We initialize our model the exact same way on all of our GPUs
3. Our optimizer has the exact same settings on all of our GPUs

Now let's focus on our training loop. The canonical one in pytorch is:

```python
loss = model(**batch)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The first 3 lines of the above can all be done asychronously. `loss.backward()` will compute gradients in each of our training processes. The clever bit is that `optimizer.step()` will synchronize the gradients across all processes before actually updating the model parameters.

So to be explicit: **`optimizer.step()` is a synchronization point across ALL processes**.

So how does pytorch achieve this?

### Splitting data across our workers - `torch.utils.data.distributed.DistributedSampler`

In our normal training script we use a `torch.utils.data.DataLoader` to batch our data. One of the arguments to DataLoader is a `sampler`, which basically samples items from the dataset when constructing the batches. You can think of the sampler as doing:

```
# simplified Sampler
worker_len = len(dataset)
random.choice(range(worker_len))
```

The clever thing that the DistributedSampler does is it partitions the length of the dataset across each of our workers. You don't even have to partition the actual dataset - it just chooses the integers that it returns from a specific subset of the dataset:

```
# simplified DistributedSampler
worker_len = len(dataset) // world_size
rank * worker_len + random.choice(range(worker_len))
```

### Gradient Synchronization - `torch.nn.parallel.DistributedDataParallel`

Funnily enough you might assume that the DDP module splits batches across processes, but that is not what it does at all!

This is a model wrapper class that sets up the gradient synchronization. I encourage you to read the documentation for this, it's very informative: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html.

This class also ensures that *model parameters are equal when you construct it*!

It achieves all of this through some very special model hooks.

## Code Changes

### Using `torchrun` instead of `python`

When training with multiple GPUs, we are actually spinning up a process for each GPU we are using. `torchrun` does this for us. It also manages the communication settings between each of the processes.

- `--standalone` argument is used when only running on a single node.
- `--nnodes` is the number of nodes we are using, in this case 1, but once we go to multiple nodes, this will be > 1.
- `--nproc-per-node` is the number of processes. `gpu` means to use all available GPUs.

#### OMP_NUM_THREADS

pytorch by default tries to take advantage of all the cores available when doing computations, even when you are on the GPU. Since we have multiple processes running pytorch, if we didn't set `OMP_NUM_THREADS` to 1, all of them would try to use all available cores.

Another way to do this is to call:

```python
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

in your training code, but torchrun will give a warning if you don't set the environment variable.

### wandb groups

While a lot of wandb usage in distributed training code seems to only initialize wandb from the main process, there's actually another way - groups! There are a ton of benefits from doing wandb in each process:
1. You can dive down into each worker's individual metrics/progress
2. Errors for each worker are logged.
