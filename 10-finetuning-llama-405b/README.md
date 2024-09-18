# Fine tuning a 405b model

## Install flash attention

```bash
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation
```

[Source](https://github.com/Dao-AILab/flash-attention)

## Loading pretrained weights

1. Using device_map auto for `transformers`
2. sync_module_states=True

## Gradient checkpointing

Modes
1. Checkpointing
2. Offload

## FSDP Prefetching

TODO

## Memory Usage