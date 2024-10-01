# Optimizing Data Loading

NOTE: This chapter's code builds off of chapter 3's code.

An important part of achieving high throughput during distributed training is ensuring that all processes are moving at roughly the same speed. If one process is much faster, it will spend a lot of time waiting for the other processes to catch up. Data loading is actually a hugely important part of this.

## Motivating Example

While writing this guide, I noticed a drop in GPU utilization **across all nodes** when moving from single node to multi node. When training single node, the GPU power draw was at 80%, and when I went to multi node, it dropped to 60% across all nodes.

It turns out data loading was consistently slower on one node, causing **all nodes** to wait for it.

In this guide's case, since data loading is relatively fast, simply updating the number of workers and the prefetch factor fixed it. In more complex examples, other optimizations or preprocessing may be needed.

## Measuring wait time

We can measure this phenomena by adding some explicit `dist.barrier()` calls in our code with our timing wrapped around it:

```diff --git a/03-multi-node/train_llm.py b/06-data-loading/train_llm.py
index d5cb05c..26cadb8 100644
--- a/03-multi-node/train_llm.py
+++ b/06-data-loading/train_llm.py
@@ -146,7 +148,10 @@ def main():
         },
     )
 
-    timers = {k: LocalTimer(device) for k in ["data", "forward", "backward", "update"]}
+    timers = {
+        k: LocalTimer(device)
+        for k in ["data", "forward", "backward", "update", "waiting"]
+    }
 
     for state["epoch"] in range(state["epoch"], args.num_epochs):
         _LOGGER.info(
@@ -168,13 +173,22 @@ def main():
                 # NOTE: for resuming
                 continue
 
+            with timers["waiting"]:
+                dist.barrier()
+
             with timers["forward"]:
                 outputs = model(**batch)
 
+            with timers["waiting"]:
+                dist.barrier()
+
             with timers["backward"]:
                 optimizer.zero_grad(set_to_none=True)
                 outputs.loss.backward()
 
+            with timers["waiting"]:
+                dist.barrier()
+
             with timers["update"]:
                 optimizer.step()
                 lr_scheduler.step()
```

## Loading data in parallel

Most slow downs in this case all come from data size:

1. If some of the processes read data more slowly, then they will already be behind. This can be due to disk reads being blocked, limits of open file descriptors, etc.
2. If you have batches of different sizes, then the model forward/backward calls will take different amounts of time.

Most of these can be handled simply by doing data loading in another process (via `num_workers` argument):

```diff --git a/03-multi-node/train_llm.py b/06-data-loading/train_llm.py
index d5cb05c..26cadb8 100644
--- a/03-multi-node/train_llm.py
+++ b/06-optimizing-data-loading/train_llm.py
@@ -88,6 +88,8 @@ def main():
         train_data,
         batch_size=args.batch_size,
         collate_fn=default_data_collator,
+        num_workers=1,
+        prefetch_factor=2,
         # NOTE: this sampler will split dataset evenly across workers
         sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
     )
```

This will cause the data loading to happen behind the scenes **in parallel to the batch processing**.

You'll need to change the num_workers and prefetch factor settings based on a number of things:
1. How big your batch size is
2. How long a single row from your dataset takes to load/preprocess
3. How fast your batches take to process

If you have `num_workers>0`, then you just want the time to fully load a batch to be less than the time to process the batch.

## Faster storage

A very common setup is to have all of your data on networked data storage. While this is convenient for our code, it is not the most efficient for data reading.

Similar to how the cache is faster than ram, and ram is faster than disk - local node storage is much faster than networked storage:

1. Cache (Fastest)
2. RAM
3. Machine local disk
4. Networked disk (Slowest)

Simply copying all of your data to each node individual can improve the speed of data loading, at the cost of more storage.
