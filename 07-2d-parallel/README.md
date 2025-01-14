# 2d parallelism (TP + DP)

Using both FSDP and TP is actually quite simple when starting from our [TP script](../06-tensor-parallel/train_llm.py).

One caveat is that this only works if you use pytorch's newer FSDP 2 api, which is still in alpha stages.

## Applying FSDP after TP

The api is much simpler than FSDP 1 api, this is all we need to add **after** our TP code:

```python
from torch.distributed._composable.fsdp import fully_shard

if mesh["dp"].size() > 1:
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh["dp"])
    fully_shard(model, mesh=mesh["dp"])
```

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

