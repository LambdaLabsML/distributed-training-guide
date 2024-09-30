# Effective Batch Size and LR

As you scale up the number of nodes, the effective batch size (the amount of items used for model updates) increases as well:

```
effective_batch_size = batch_size * world_size
```

As you may know, increasing the batch size means that the variance of the data that your model is training on decreases, meaning your gradients will be much smoother. This directly impacts the dynamics of how your model learns and changes!

If you want to **exactly match the dynamics of single gpu training** when moving to multi node training, this chapter is aimed at you!

## Scaling Rules

If you want exact training dynamics, you have to also scale the learning rate. However, this depends on what optimizer you are using. The exact rules are not fully understood, and you can look into the following papers for more information:

- [Exploring Learning Rate Scaling Rules for Distributed ML Training on Transient Resources](https://anakli.inf.ethz.ch/papers/learning_rate_distribml22.pdf)

As of writing this, the most common rules that people use to scale learning rate are:

### Linear scaling rule

```python
lr = args.lr * dist.get_world_size()
```

This was first reported in the large minibatch SGD paper above. However this doesn't quite produce exactly the same training dynamics, and the paper actually used a **factor of the world size**.

NOTE: **Be careful when using this for optimizers other than SGD**

References:
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677)

### Square root scaling rule

```python
lr = args.lr * numpy.sqrt(dist.get_world_size())
```

This is proposed for use with the Adam optimizer, and maintains the square root of the variance of the gradient when scaling the number of batches.

References:
- [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/pdf/1404.5997)
- [Large-Batch Training for LSTM and Beyond](https://arxiv.org/pdf/1901.08256)

## Code Changes

```diff --git a/03-multi-node/train_llm.py b/95-effective-batch-size-and-lr/train_llm.py
index 38f3cf0..0cd2fac 100644
--- a/03-multi-node/train_llm.py
+++ b/95-effective-batch-size-and-lr/train_llm.py
@@ -89,9 +89,16 @@ def main():
     )
     _LOGGER.info(f"{len(dataloader)} batches per epoch")
 
-    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
+    if args.lr_scaling == "static":
+        lr = args.lr
+    elif args.lr_scaling == "linear":
+        lr = args.lr * world_size
+    elif args.lr_scaling == "sqrt":
+        lr = args.lr * numpy.sqrt(world_size)
+
+    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
-        optimizer, T_max=1000, eta_min=args.lr * 1e-2
+        optimizer, T_max=1000, eta_min=lr * 1e-2
     )
@@ -305,6 +312,9 @@ def _get_parser() -> argparse.ArgumentParser:
     parser.add_argument("--log-freq", default=100, type=int)
     parser.add_argument("--ckpt-freq", default=500, type=int)
     parser.add_argument("--dataset-cache-root", default="../.cache")
+    parser.add_argument(
+        "--lr-scaling", default="static", choices=["static", "linear", "sqrt"]
+    )
     return parser
```