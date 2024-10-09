# Gradient Accumulation

Gradient accumulation is a way to increase the effective batch sizes of your model updates.

It is normally applied when your model is so big that you use a lower batch size when running the forward/backward pass.

If on a single GPU you have a batch size of 4, and a gradient accumulation of 2, then your effective batch size is 8. 

However, applying gradient accumulation in a standard way will cause slowdowns in distributed training setting because of gradient synchronization.

## Standard Implementation

```python
outputs = model(**batch)
outputs.loss.backward()
if i_step % grad_accum == 0:
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
```

## DataDistributedParalell Implementation

In a distributed setting, gradients will be synchronized at multiple points during our forward pass. It turns out we need to delay this synchronization until we do the full model step!

We can use [torch.nn.parallel.DistributedDataParallel.no_sync](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync) for this:

```python
from contextlib import nullcontext
maybe_sync_grads = model.no_sync if i_step % grad_accum != 0 else nullcontext
with maybe_sync_grads():
    outputs = model(**batch)
    outputs.loss.backward()
if i_step % grad_accum == 0:
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
```
