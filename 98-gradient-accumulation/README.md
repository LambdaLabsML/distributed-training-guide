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
    optimizer.zero_grad()
```

## DataDistributedParalell Implementation

In a distributed setting, gradients will be synchronized at multiple points during our forward pass. It turns out we need to delay this synchronization until we do the full model step!

We can use [torch.nn.parallel.DistributedDataParallel.no_sync](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync) for this:

```python
from contextlib import nullcontext
grad_sync = model.no_sync if i_step % grad_accum != 0 else nullcontext
with grad_sync():
    outputs = model(**batch)
    outputs.loss.backward()
if i_step % grad_accum == 0:
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
```

## Our final implementation

```diff
--- a/03-multi-node/train_llm.py
+++ b/98-gradient-accumulation/train_llm.py
@@ -1,4 +1,5 @@
 import argparse
+from contextlib import nullcontext
 from itertools import chain
 import json
 import multiprocessing
@@ -164,16 +165,22 @@ def main():
                 # NOTE: for resuming
                 continue
 
-            with timers["forward"]:
+            if i_step % args.grad_accum == 0:
+                maybe_sync_grads = nullcontext
+            else:
+                maybe_sync_grads = model.no_sync
+
+            with timers["forward"], maybe_sync_grads():
                 outputs = model(**batch)
 
-            with timers["backward"]:
-                optimizer.zero_grad()
+            with timers["backward"], maybe_sync_grads():
                 outputs.loss.backward()
 
             with timers["update"]:
-                optimizer.step()
-                lr_scheduler.step()
+                if i_step % args.grad_accum == 0:
+                    optimizer.step()
+                    lr_scheduler.step()
+                    optimizer.zero_grad()
```