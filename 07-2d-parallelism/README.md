# Training a 405B model with 2d Parallelism

So far we've just been using data parallel techniques. You may have heard of other parallelism techniques, and indeed the [Llama 405B paper](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) actually uses 4D parallelism when training the 405B model:

1. Data parallel (FSDP as we've learned)
2. Tensor parallel (this chapter)
3. Context parallel (For long context lengths)
4. Pipeline/model parallel

In this chapter we are going to be adding tensor parallelism to our 405b training script from [chapter 6](../06-training-llama-405b/) to see what improvements we get.

## Basics: What is tensor parallelism?

Some operations can be easily split among multiple workers without having to share any parameters. If you remember with fully sharded data parallel, we still have to gather all the weights onto each node. That is due to the way the operations we were wrapping worked.

Let's think about applying a ReLU Activation to a tensor at first - well we could easily split the input tensor across our workers and all the workers could separately apply the ReLU function to each of their parts.

Tensor Parallelism is most often applied with Linear layers, which occur a lot in Transformer architecture. Basically the weight & bias matrix in the Linear layer are split up across your workers, and then when applying the forward to the tensor, only the tensor needs to be split across the workers, not the weights.

This ends up being much cheaper/faster!

## Pytorch Reference Docs

For completeness here are the relevant docs/guides from pytorch on how to achieve this:
- [API docs](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#tensor-parallelism-torch-distributed-tensor-parallel)
- [2d Parallelism Tutorial](https://pytorch.org/tutorials/intermediate/TP_tutorial.html#large-scale-transformer-model-training-with-tensor-parallel-tp)
- [Device Mesh tutorial](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)

## Implementation

There are a couple of parts to implementing this. For starters we are going to need what pytorch calls a [Device Mesh](https://pytorch.org/docs/stable/distributed.html#torch.distributed.device_mesh.DeviceMesh). Why? Because this will let us split workers into groups based on what node they are on.

For a high level view first, we are going to:

1. Internal to each node do tensor parallelism (so split Linear layers up across the node)
2. Across nodes do data parallelism (what we've been doing with FSDP so far)

A DeviceMesh helps us work with this.

```python
# NOTE: assumes all nodes have the same number of gpus
num_local_gpus = torch.cuda.device_count()
num_nodes = dist.get_world_size() // num_local_gpus
mesh = dist.device_mesh.init_device_mesh("cuda", (num_nodes, num_local_gpus), ("dp", "tp"))
```

### Model Architecture Reference

For reference for the next couple of sections, here is the architecture for 405B:

```python
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 16384)
    (layers): ModuleList(
      (0-125): 126 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=16384, out_features=16384, bias=False)
          (k_proj): Linear(in_features=16384, out_features=1024, bias=False)
          (v_proj): Linear(in_features=16384, out_features=1024, bias=False)
          (o_proj): Linear(in_features=16384, out_features=16384, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=16384, out_features=53248, bias=False)
          (up_proj): Linear(in_features=16384, out_features=53248, bias=False)
          (down_proj): Linear(in_features=53248, out_features=16384, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((16384,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((16384,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((16384,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=16384, out_features=128256, bias=False)
)
```

TODO we apply FSDP to the decoder layer block, but need to apply TP to the linear layers/MLP?

### Applying tensor parallelism to our model

Note that this is done *before* passing our model to FSDP. Luckily this all works very seamlessly with meta models.

```python
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
```
TODO


### Using our mesh with FSDP

For this, we can just pass our dp mesh directly to the FSDP constructor:

```python
model = FSDP(..., device_mesh=mesh["dp"])
```

Note that this allows us to try other versions of sharding.

You can also just pass the entire mesh into device_mesh. TODO what should be done??

Here are three options:
1. `sharding_strategy=ShardingStrategy.FULL_SHARD` and `device_mesh=mesh["dp"]`

TODO

2. `sharding_strategy=ShardingStrategy.SHARD_GRAD_OP` and `device_mesh=mesh`

TODO

3. `sharding_strategy=ShardingStrategy.HYBRID_SHARD` and `device_mesh=mesh`

TODO

4. `sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2` and `device_mesh=mesh`

TODO


## Questions

Does tp.SequenceParallel apply the sharding for you?
How does RowParallelism work with Embedding? Won't shards not have access to the correct indices? Why Not ColParallelism?