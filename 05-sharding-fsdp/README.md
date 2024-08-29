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
