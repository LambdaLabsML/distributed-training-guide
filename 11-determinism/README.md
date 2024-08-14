# Determinism across resumes

See pytorch's documnetation on reproducibility: https://pytorch.org/docs/stable/notes/randomness.html#reproducibility

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCHELASTIC_ERROR_FILE=./error.json OMP_NUM_THREADS=1 torchrun \
    --rdzv-id multi-node \
    --rdzv-backend c10d \
    --rdzv-endpoint <IP ADDRESS of main node>:<port> \
    --nnodes 2 \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ./logs \
    train_llm.py \
    --experiment-name multi-node \
    --dataset-name Skylion007/openwebtext \
    --model-name openai-community/gpt2 \
    --batch-size 64
```

Notably we are also saving & restoring the rng states from various libraries, and explicitly seeding the workers for data loading.

## Code Changes

```diff
diff --git a/03-multi-node/train_llm.py b/10-determinism/train_llm.py
index 24eacbd..0a3a029 100644
--- a/03-multi-node/train_llm.py
+++ b/10-determinism/train_llm.py
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
 
@@ -116,6 +121,13 @@ def main():
         lr_scheduler.load_state_dict(_load_to_device(exp_dir / "lr_scheduler.pt"))
         with open(exp_dir / "state.json") as fp:
             state = json.load(fp)
+        rng_state = torch.load(
+            exp_dir / "rng.pt", weights_only=False, map_location="cpu"
+        )
+        numpy.random.set_state(rng_state["np"])
+        random.setstate(rng_state["random"])
+        torch.set_rng_state(rng_state["torch"])
+        torch.cuda.set_rng_state(rng_state["cuda"][local_rank], device)
         resumed = True
     _LOGGER.info(f"[{rank}] Resumed={resumed} | {state}")
 
@@ -208,11 +220,26 @@ def main():
                     torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
                     with open(exp_dir / "state.json", "w") as fp:
                         json.dump(state, fp)
+                    torch.save(
+                        {
+                            "np": numpy.random.get_state(),
+                            "random": random.getstate(),
+                            "torch": torch.get_rng_state(),
+                            "cuda": torch.cuda.get_rng_state_all(),
+                        },
+                        exp_dir / "rng.pt",
+                    )
                 dist.barrier()
 
         state["epoch_step"] = 0
 
 
+def _seed_worker(worker_id):
+    worker_seed = torch.initial_seed() % 2**32
+    numpy.random.seed(worker_seed)
+    random.seed(worker_seed)
+
+
 def _load_and_preprocess_data(args, tokenizer, config):
     data = datasets.load_dataset(
         args.dataset_name, trust_remote_code=True, cache_dir=args.dataset_cache_root
```