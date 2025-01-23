# 2d parallelism (TP + DP)

Using both [FSDP](../04-fully-sharded-data-parallel) and [TP](../06-tensor-parallel) is actually quite simple code wise when starting from our [chapter 6 TP script](../06-tensor-parallel/train_llm.py).

**Disclaimer** this only works if you use pytorch's **newer FSDP 2 api, which is still in alpha stages**.

What does using these two together mean exactly? Let's get into an example with 6 GPUs, 2 way FSDP and 3 way TP:

<img width="941" alt="image" src="https://github.com/user-attachments/assets/4c171de9-6a41-4bae-9278-6abe81ee5c84" />

When we first start out every gpu holds the full model. Then we shard the model into 3 pieces (our TP dimension). Note that GPU 0 and GPU 3 **have the exact same shard**! This is because they are the same tensor parallel rank, but are different data parallel ranks. This means we have **duplicated** our model across our data parallel dimension.

When we apply FSDP in the next step, we split those duplicated shards! So Shard 0 (which is duplicated on GPU 0 & 3) is split into two pieces (Shard 0,a and Shard 0,b).

By the end we have 6 distinct shards of our model split on every GPU.

Now if you remember with FSDP, it does an allgather of all the shards before the forward pass. When GPU 0 & GPU 3 are executing their forward passes, they will gather the two shards (Shard 0,a and Shard 0,b) into local memory to form Shard 0, so that each one can use the full shard during computation.

## Applying FSDP after TP

We are starting from our [chapter 6 code](../06-tensor-parallel/train_llm.py), which already support TP. So we just need to add FSDP to the script:

The api is much simpler than FSDP 1 api, this is all we need to add **after** our TP code:

```python
from torch.distributed._composable.fsdp import fully_shard

if mesh["dp"].size() > 1:
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh["dp"])
    fully_shard(model, mesh=mesh["dp"])
```

Note how we are passing our `mesh["dp"]` here to indicate that this is happening across our data parallel dimension.

## Controlling TP size

When creating our mesh we are going to set the TP size based on a CLI argument:

```python
assert world_size % args.tp == 0

mesh = dist.device_mesh.init_device_mesh(
    "cuda",
    (world_size // args.tp, args.tp),
    mesh_dim_names=("dp", "tp"),
)
```

and add it to our argparser:

```python
parser.add_argument("--tp", default=8, type=int)
```

## Performance with different configurations

Here are some training results for 4 different setups of the TP size:
- 1x8 is 8 way TP, and no data parallelism. `--batch-size 18 --tp 8`
- 2x4 is 4 way TP, with 2 groups of FSDP. `--batch-size 14 --tp 4`
- 4x2 is 2 way TP, with 4 groups of FSDP. `--batch-size 10 --tp 2`
- 8x1 is FSDP. `--batch-size 7 --tp 1`

Note that all of these runs have the same `--lr` while having different batch sizes, which is why the loss curves are slightly different.

<img src="https://github.com/user-attachments/assets/8645b7d2-992f-4f49-9214-f6c5d4d42c37" width="480px" />

<img src="https://github.com/user-attachments/assets/1b9269ce-c1db-43e4-9fd7-0bbf11871b11" width="480px" />

