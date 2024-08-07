# Determinism across resumes

See pytorch's documnetation on reproducibility: https://pytorch.org/docs/stable/notes/randomness.html#reproducibility

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCHELASTIC_ERROR_FILE=./error.json OMP_NUM_THREADS=1 torchrun \
    --rdzv-id multi-node \
    --rdzv-backend c10d \
    --rdzv-endpoint <IP ADDRESS of main node>:<port> \
    --nnodes 2 \
    --nproc-per-node gpu \
    train_llm.py \
    --experiment-name multi-node \
    --dataset-name Skylion007/openwebtext \
    --model-name openai-community/gpt2 \
    --batch-size 64
```

Notably we are also saving & restoring the rng states from various libraries, and explicitly seeding the workers for data loading.

## Code Changes

```diff
diff --git a/03-multi-node/train_llm.py b/09-determinism/train_llm.py
index e593b16..759c2cc 100644
--- a/03-multi-node/train_llm.py
+++ b/09-determinism/train_llm.py
@@ -40,6 +40,7 @@ def main():
 
     torch.set_num_threads(1)
     torch.set_num_interop_threads(1)
+    torch.use_deterministic_algorithms(True)
 
     torch.manual_seed(args.seed)
     torch.cuda.manual_seed_all(args.seed)
@@ -84,6 +85,8 @@ def main():
         train_data = _load_and_preprocess_data(args, tokenizer, config)
     _LOGGER.info(f"[{rank}] {len(train_data)} training samples")
 
+    g = torch.Generator()
+    g.manual_seed(args.seed)
     dataloader = DataLoader(
         train_data,
         batch_size=args.batch_size,
@@ -91,6 +94,8 @@ def main():
         num_workers=1,
         # NOTE: this sampler will split dataset evenly across workers
         sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
+        worker_init_fn=_seed_worker,
+        generator=g,
     )
     _LOGGER.info(f"[{rank}] {len(dataloader)} batches per epoch")
 
@@ -107,6 +112,12 @@ def main():
         "global_step": 0,
         "epoch_step": 0,
         "running_loss": 0,
+        "rng": {
+            "np": numpy.random.get_state(),
+            "random": random.getstate(),
+            "torch": torch.get_rng_state(),
+            "cuda": torch.cuda.get_rng_state_all(),
+        },
     }
 
     resumed = False
@@ -116,6 +127,10 @@ def main():
         lr_scheduler.load_state_dict(_load_to_device(exp_dir / "lr_scheduler.pt"))
         with open(exp_dir / "state.json") as fp:
             state = json.load(fp)
+            numpy.random.set_state(state["rng"]["np"])
+            random.setstate(state["rng"]["random"])
+            torch.set_rng_state(state["rng"]["torch"])
+            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
         resumed = True
     _LOGGER.info(f"[{rank}] Resumed={resumed} | {state}")
 
@@ -206,6 +221,10 @@ def main():
                     torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
                     torch.save(model.state_dict(), exp_dir / "model.pt")
                     torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
+                    state["rng"]["np"] = numpy.random.get_state()
+                    state["rng"]["random"] = random.getstate()
+                    state["rng"]["torch"] = torch.get_rng_state()
+                    state["rng"]["cuda"] = torch.cuda.get_rng_state_all()
                     with open(exp_dir / "state.json", "w") as fp:
                         json.dump(state, fp)
                 dist.barrier()
@@ -213,6 +232,12 @@ def main():
         state["epoch_step"] = 0
 
 
+def _seed_worker(worker_id):
+    worker_seed = torch.initial_seed() % 2**32
+    numpy.random.seed(worker_seed)
+    random.seed(worker_seed)
```