# Sharding Across GPUs

Up to this point we have assumed that both the model & optimizer fully fit on a single GPU. So each GPU during our training process fully contains a copy of the model and optimizer.

This becomes an issue once the model becomes big enough - either the model itself cannot fit, or the optimizer (which usually contains 1-4x the memory of the model) cannot fit anymore.

Sharding refers to the idea of only keeping a small part (or shard) of the model or optimizer on a single GPU. This obviously needs some extra synchronization between the workers as not all workers would have the full state.

## PyTorch FullyShardedDataParallel (FSDP)
