# Minimizing Synchronization Time

As we have learned in previous sections, there are some explicit synchronization points while using pytorch DDP/DistributedSampler.

An important part of achieving high throughput during distributed training is ensuring that all processes are moving at roughly the same speed.

If one process is much faster, it will spend a lot of time waiting for the other processes to catch up.

## Measuring wait time

We can measure this phenomena by adding some explicit `dist.barrier()` calls in our code with our timing wrapped around it:


```diff
diff --git a/03-multi-node/train_llm.py b/05-minimizing-synchronization-time/train_llm.py
index e593b16..35eb66f 100644
--- a/03-multi-node/train_llm.py
+++ b/05-minimizing-synchronization-time/train_llm.py
with timers["data"], torch.no_grad():
    batch = next(batch_iter)
    batch = {k: v.to(device=device) for k, v in batch.items()}

+ with timers["waiting"]:
+     dist.barrier()

with timers["forward"]:
    outputs = model(**batch)

+ with timers["waiting"]:
+     dist.barrier()

with timers["backward"]:
    optimizer.zero_grad()
    outputs.loss.backward()

+ with timers["waiting"]:
+     dist.barrier()

with timers["update"]:
    optimizer.step()
    lr_scheduler.step()
```

## Causes of slowdowns

Most slow downs in this case all come from data size:

1. If some of the processes read data more slowly, then they will already be behind. This can be due to disk reads being blocked, limits of open file descriptors, etc.
2. If you have batches of different sizes, then the model forward/backward calls will take different amounts of time.

Most of these can be handled simply by doing data loading in another process (via `num_workers` argument):

```diff
diff --git a/03-multi-node/train_llm.py b/05-minimizing-synchronization-time/train_llm.py
index e593b16..35eb66f 100644
--- a/03-multi-node/train_llm.py
+++ b/05-minimizing-synchronization-time/train_llm.py
dataloader = DataLoader(
    train_data,
    batch_size=args.batch_size,
    collate_fn=default_data_collator,
-    num_workers=0,
+    num_workers=1,
-    prefetch_factor=None,
+    prefetch_factor=2,
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
