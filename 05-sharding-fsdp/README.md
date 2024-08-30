# Sharding Across GPUs

Up to this point we have assumed that both the model & optimizer fully fit on a single GPU. So each GPU during our training process fully contains a copy of the model and optimizer.

This becomes an issue once the model becomes big enough - either the model itself cannot fit, or the optimizer (which usually contains 1-4x the memory of the model) cannot fit anymore.

Sharding refers to spreading the **storage** of a combination of: optimizer state, gradients, and/or model parameters **across your GPUs**. **The execution of layers DOES NOT CHANGE**.

What this means:

1. Each layer of your model still needs to pull the **entire** layer's parameters/gradients/optimizer states into GPU memory. After the layer is done, then those pieces are resharded.
2. There are synchronization costs to un-shard and re-shard before and after each layer.
3. Sharding does not reduce the peak memory cost of your biggest layer.

**Sharding is a data parallel technique! NOT a model/tensor/pipeline parallel technique**

## PyTorch FullyShardedDataParallel (FSDP)

References:
- [FSDP Docs](https://pytorch.org/docs/stable/fsdp.html)
- [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html).

### Initialization **after** sharding - the `meta` device


[meta device docs](https://pytorch.org/docs/stable/meta.html)

This is useful because the meta device does not allocate any memory at all! It makes model initialization extremely fast. You can actually run a lot of the ops with the meta device as well, only the shape will be modified.

This is useful when training large models when we don't want to actually initialize the fully model in memory on each device. We can then use this meta model in conjuction with the FSDP constructor to only initialize the model weights **after** the model has been sharded across the GPUs.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
```

### The FSDP Constructor

Here is our FSDP constructor, let's explore each of these arguments in more detail. ALL of the options here have an impact on throughput, memory usage, and peak memory usage.

```python
model = FullyShardedDataParallel(
    model,
    device_id=local_rank,
    param_init_fn=safe_param_init_fn,
    sync_module_states=True,
    auto_wrap_policy=wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(offload_params=args.cpu_offload == "on"),
    backward_prefetch=getattr(BackwardPrefetch, args.bwd_prefetch, default=None),
)
```

#### Parameter initialization (when using the `meta` device) - `param_init_fn`

TODO

#### sync_module_states

TODO

#### What layers to shard - the `auto_wrap_policy`

TODO

#### What to shard - `sharding_strategy`

[ShardingStrategy docs](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy)

FSDP fully implements everything you can do with deepspeed! Here's how the stages align:

| FSDP ShardingStrategy | DeepSpeed ZeRO Stage | Shard Optimizer states | Shard Gradients | Shard Parameters |
| --------------------- | -------------------- | ---------------------- | --------------- | ---------------- |
| `FULL_SHARD`          | 3                    | ✅                      | ✅               | ✅                |
| `SHARD_GRAD_OP`       | 2                    | ✅                      | ✅               | ❌                |
| `NO_SHARD`  (DDP)     | 0                    | ❌                      | ❌               | ❌                |
| `HYBRID_SHARD`        | ZeRO++ 3             | ✅ (intra-node)         | ✅ (intra-node)  | ✅ (intra-node)   |


#### CPU Offload

[CPUOffload docs](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.CPUOffload)

> If True, then this offloads gradients to CPU as well, meaning that the optimizer step runs on CPU.

This option **heavily** reduces memory requirements - at the cost of a lot of compute and memory bandwidth. The forward & backward pass runs on the GPU, then gradients are offloaded to CPU and the optimizer runs on the CPU.

Note that this option will **NOT reduce peak GPU memory requirements** - each layer will still be fully executed in the GPU.

#### Prefetching layer weights

[BackwardPrefetch docs](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.BackwardPrefetch)

These options are mainly for tuning **peak** memory usage vs throughput.

| BackwardPrefetch | Peak Memory Usage per Layer                                                     |
| ---------------- | ------------------------------------------------------------------------------- |
| BACKWARD_PRE     | current set of parameters, next set of parameters, and current set of gradients |
| BACKWARD_POST    | next set of parameters, and current set of gradients                            |
| None             | -                                                                               |

### Sharded Checkpoints

TODO

#### Saving a sharded checkpoint

TODO

#### Loading a sharded checkpoint

TODO

#### Converting a sharded checkpoint to a full state dict checkpoint

TODO

### Run Command

Same command as normal:

```bash
cd distributed-training-guide/05-sharding-fsdp
TORCHELASTIC_ERROR_FILE=../error.json OMP_NUM_THREADS=1 torchrun --standalone \
    --nnodes 1 \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ../logs \
    train_llm.py \
    --experiment-name fsdp \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2 \
    --batch-size 64 \
    --cpu-offload on
```

### Examples of memory usage with different configurations

* `peak GPU memory`: The highest GPU memory allocated *at any point* during a single training loop iteration (*during* forward/backward/step)
* `valley GPU memory`: The GPU memory allocated at the end of a single training loop iteration (*after* forward/backward/step)

#### meta-llama/Llama-2-7B-hf

| GPUs          | --numel-to-wrap | --cpu-offload | --batch-size | valley/peak GPU memory (per GPU) |
| ------------- | --------------- | ------------- | ------------ | -------------------------------- |
| 8xA100 (80GB) | 100_000_000     | off           | 10           | 8GB / 74GB                       |
| 8xA100 (80GB) | 100_000_000     | **on**        | 10           | 1.4GB / 68.7GB                   |

<details>
    <summary>Wrapped model architecture</summary>
    
    ```python
    FullyShardedDataParallel(
      (_fsdp_wrapped_module): LlamaForCausalLM(
        (model): FullyShardedDataParallel(
          (_fsdp_wrapped_module): LlamaModel(
            (embed_tokens): FullyShardedDataParallel(
              (_fsdp_wrapped_module): Embedding(32000, 4096)
            )
            (layers): ModuleList(
              (0-31): 32 x LlamaDecoderLayer(
                (self_attn): LlamaSdpaAttention(
                  (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
                  (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
                  (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
                  (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
                  (rotary_emb): LlamaRotaryEmbedding()
                )
                (mlp): FullyShardedDataParallel(
                  (_fsdp_wrapped_module): LlamaMLP(
                    (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
                    (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
                    (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
                    (act_fn): SiLU()
                  )
                )
                (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
                (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
              )
            )
            (norm): LlamaRMSNorm((4096,), eps=1e-05)
            (rotary_emb): LlamaRotaryEmbedding()
          )
        )
        (lm_head): FullyShardedDataParallel(
          (_fsdp_wrapped_module): Linear(in_features=4096, out_features=32000, bias=False)
        )
      )
    )
    ```
</details>

#### meta-llama/Llama-2-70B-hf

We actually **need** to use `--cpu-offload on` in this case - we can fit the model in memory, but the forward/backward/step passes don't fit, even with batch size 1.

| GPUs          | --numel-to-wrap | --cpu-offload | --batch-size | valley/peak GPU memory (per GPU) |
| ------------- | --------------- | ------------- | ------------ | -------------------------------- |
| 8xA100 (80GB) | 100_000_000     | on            | 2            | 0.3GB / 72.3 GB                  |


<details>
    <summary>Wrapped model architecture</summary>

    ```python
    FullyShardedDataParallel(
      (_fsdp_wrapped_module): LlamaForCausalLM(
        (model): LlamaModel(
          (embed_tokens): FullyShardedDataParallel(
            (_fsdp_wrapped_module): Embedding(32000, 8192)
          )
          (layers): ModuleList(
            (0-79): 80 x LlamaDecoderLayer(
              (self_attn): FullyShardedDataParallel(
                (_fsdp_wrapped_module): LlamaSdpaAttention(
                  (q_proj): Linear(in_features=8192, out_features=8192, bias=False)
                  (k_proj): Linear(in_features=8192, out_features=1024, bias=False)
                  (v_proj): Linear(in_features=8192, out_features=1024, bias=False)
                  (o_proj): Linear(in_features=8192, out_features=8192, bias=False)
                  (rotary_emb): LlamaRotaryEmbedding()
                )
              )
              (mlp): LlamaMLP(
                (gate_proj): FullyShardedDataParallel(
                  (_fsdp_wrapped_module): Linear(in_features=8192, out_features=28672, bias=False)
                )
                (up_proj): FullyShardedDataParallel(
                  (_fsdp_wrapped_module): Linear(in_features=8192, out_features=28672, bias=False)
                )
                (down_proj): FullyShardedDataParallel(
                  (_fsdp_wrapped_module): Linear(in_features=28672, out_features=8192, bias=False)
                )
                (act_fn): SiLU()
              )
              (input_layernorm): LlamaRMSNorm((8192,), eps=1e-05)
              (post_attention_layernorm): LlamaRMSNorm((8192,), eps=1e-05)
            )
          )
          (norm): LlamaRMSNorm((8192,), eps=1e-05)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (lm_head): FullyShardedDataParallel(
          (_fsdp_wrapped_module): Linear(in_features=8192, out_features=32000, bias=False)
        )
      )
    )
    ```
</details>
