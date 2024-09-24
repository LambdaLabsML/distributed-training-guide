# Fine tuning a 405b model

## Use flash attention

```bash
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation
```

[Source](https://github.com/Dao-AILab/flash-attention)

```python
model = AutoModelForCausalLM.from_pretrained(
    ...
    attn_implementation="flash_attention_2",
)
```

## Install accelerate

TODO might not be necessary

For device_map="auto" use in transformers.

```bash
pip install accelerate==0.34.2
```

## Download model weights

The weights need to be on a shared network drive accessible to all nodes. This download script assumes that this repo is cloned into shared network drive.

```bash
cd distributed-training-guide/10-finetuning-llama-405b
export HF_HOME=<your shared network drive>
python download.py
```

The download script will default HF_HOME to `distributed-training-guide/.cache` if not specified.

## Loading pretrained weights

1. Using device_map "cpu" for `transformers`
2. sync_module_states=True

We can't actually use device_map "auto", because this will fully utilize the rank 0 gpu. When we try to initialize FSDP later we won't have any memory left to allocate. Instead we use device_map="cpu":

```python
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=dtype,
    # NOTE: only load the weights on rank 0
    #       these will be sent to other ranks
    #       with `sync_module_states=True` later
    device_map="cpu" if rank == 0 else "meta",
    attn_implementation="flash_attention_2",
)
```

## Gradient checkpointing

Modes
1. Checkpointing
2. Offload

## FSDP Prefetching

TODO

## Launch command

We provide a customized launch.sh script here based on the bash command for spawning torchrun on all available nodes:

```bash
cd distributed-training-guide/10-finetuning-llama-405b
vim hosts # NOTE: put each host on a different line in this file
bash launch.sh
```

Also note that this launch.sh specifies `HF_HOME` as an environment variable in the tmux session, so if you've not used the default value of `distributed-training-guide/.cache`, please update the script!

## Monitoring Logs

Here's a convenience bash command for tailing all torchrun log files at once:

```bash
find ../logs/ -name \*.log | xargs tail -f
```

## Memory Usage

