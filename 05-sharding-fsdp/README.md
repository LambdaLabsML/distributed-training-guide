# Sharding Across GPUs

Up to this point we have assumed that both the model & optimizer fully fit on a single GPU. So each GPU during our training process fully contains a copy of the model and optimizer.

This becomes an issue once the model becomes big enough - either the model itself cannot fit, or the optimizer (which usually contains 1-4x the memory of the model) cannot fit anymore.

Sharding refers to the idea of only keeping a small part (or shard) of the model or optimizer on a single GPU. This obviously needs some extra synchronization between the workers as not all workers would have the full state.

## PyTorch FullyShardedDataParallel (FSDP)

See official [FSDP Docs](https://pytorch.org/docs/stable/fsdp.html) & [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html).

> ### Aside: Relation to DeepSpeed
> 
> FSDP fully implements everything you can do with deepspeed! Here's how the stages align:
> 
> - `ShardingStrategy.FULL_SHARD` maps to the DeepSpeed ZeRO Stage-3. Shards optimizer states, gradients and parameters.
> - `ShardingStrategy.SHARD_GRAD_OP` maps to the DeepSpeed ZeRO Stage-2. Shards optimizer states and gradients.
> - `ShardingStrategy.NO_SHARD` maps to ZeRO Stage-0. No sharding wherein each GPU has full copy of model, optimizer states and gradients.
> - `ShardingStrategy.HYBRID_SHARD` maps to ZeRO++ Stage-3 wherein zero_hpz_partition_size=<num_gpus_per_node>. Here, this will shard optimizer states, gradients and parameters within each node while each node has full copy.

### Code Changes

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
    --batch-size 64
```

### Examples of memory usage with different configurations

* `peak GPU memory`: The highest GPU memory allocated *at any point* during a single training loop iteration (*during* forward/backward/step)
* `valley GPU memory`: The GPU memory allocated at the end of a single training loop iteration (*after* forward/backward/step)

#### meta-llama/Llama-2-7B-hf

| GPUs          | --numel-to-wrap | --cpu-offload | --batch-size | valley/peak GPU memory (per GPU) |
| ------------- | --------------- | ------------- | ------------ | -------------------------------- |
| 8xA100 (80GB) | 100_000_000     | off           | 10           | 8GB / 74GB                       |
| 8xA100 (80GB) | 100_000_000     | **on**        | 10           | 1.4GB / 68.7GB                   |

###### Wrapped model architecture
<details>
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


###### Wrapped model architecture
<details>
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