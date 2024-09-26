# Training a 405b model

Here we are going to utilize a huge cluster to train Llama 3.1 405B. **This does not utilize LORA!** We are actually fully training the weights of a 405b model in plain pytorch.

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

## Download model weights

There are two options here:

1. A shared network drive
2. Locally on each node

Node local storage is **vastly** faster. For some numbers, while running this script on 8 8xH100 80GB nodes, the shared network drive took 50 minutes to initialize, while the node local storage only took 3 minutes.

There's a download script in this repo for utility, run this on node 0:

```bash
cd distributed-training-guide/10-finetuning-llama-405b
python download.py
```

## Loading pretrained weights

There's three parts to this:

1. Using device_map "cpu" for rank 0, and meta device for rank > 0
2. Using from_config instead of from_pretrained on rank > 0
3. FSDP.sync_module_states=True

We can't actually use device_map "auto", because this will fully utilize the rank 0 gpu. When we try to initialize FSDP later we won't have any memory left to allocate. Instead we use device_map="cpu" on rank 0:

```python
if rank == 0:
    model = AutoModelForCausalLM.from_pretrained(..., device_map="cpu")
else:
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
```

Then later, sync_module_states in FSDP constructor will make sure the weights are broadcasted from rank 0 to the other ranks.

## Sharding the LlamaDecoderLayer

Here we are just going to be sharding the LlamaDecoderLayer (there's 191 of them). We can use the `transformer_auto_wrap_policy` to target the specific class for that layer:

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

wrap_policy = functools.partial(
    transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer}
)

FSDP(..., auto_wrap_policy=wrap_policy)
```

## Gradient checkpointing

Another piece of this is gradient checkpointing, which saves a lot of memory. This piece of code has to go **after** the FSDP constructor!!! I'm not exactly sure of the reason, but it doesn't work before the FSDP initialization.

The method we are using is kind of a hidden method in pytorch, but this is actually exactly what [accelerate uses under the hood](https://github.com/huggingface/accelerate/blob/v0.34.2/src/accelerate/accelerator.py#L1492) so rest assured that it is a "standard" way of doing it:

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

apply_activation_checkpointing(
    model, checkpoint_wrapper_fn=checkpoint_wrapper, auto_wrap_policy=wrap_policy
)
```

## Launch command

We provide a customized launch.sh script here based on the bash command for spawning torchrun on all available nodes:

```bash
cd distributed-training-guide/10-finetuning-llama-405b
bash launch.sh # NOTE: this is non blocking
```

Also note that this launch.sh specifies `HF_HOME` as an environment variable in the tmux session, so if you've not used the default value of `/home/ubuntu/.cache/huggingface`, please update the script!

You can change the hostnames in the `hosts` file in this directory.

## Monitoring

The log files are really useful for monitoring the progress of everything. Here's a bash command for tailing all of them at once:

```bash
find ../logs/ -name \*stderr.log | xargs tail -f
```

Additionally, we have a top like utility script for monitoring the entire cluster at the top level of this directory:

```bash
python ../top-cluster.py hosts
```

## Run statistics

### Memory Usage

### Throughput
