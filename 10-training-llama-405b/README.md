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
cd distributed-training-guide/10-training-llama-405b
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
    with torch.device("cpu"):
        model = AutoModelForCausalLM.from_pretrained(...)
else:
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
```

Then later, sync_module_states in FSDP constructor will make sure the weights are broadcasted from rank 0 to the other ranks.

## Sharding Llama 405b

Most of the tutorials on training Llama 405b just shard the LlamaDecoderLayer (there's 191 of them). However during testing I also found that sharding the nn.Embedding layer at the beginning of the network improved throughput and reduced memory usage. We can use the `transformer_auto_wrap_policy` to target the specific class for those layer:

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer, nn.Embedding},
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

## fused Optimizer implementation

When using CPUOffload feature of FSDP, the optimizer entirely runs on the CPU. This is because there is significant cost to transfer data to and from the GPU when doing optimizer.step(). At the time of this being written there are open issues on how to overlap the optimizer.step() with the next forward() call.

By default the optimizers will use a per tensor forward call on the cpu, but there are flags you can enable to get a bit of a speedup:

```python
torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
```

## zero_grad(set_to_none=???)

You may have seen this set_to_none argument in [optimizer.zero_grad()](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html). According to the docs:

> This will in general have lower memory footprint, and can modestly improve performance.

Basically set_to_none=True will deallocate the gradients after they are used. In cases where we aren't memory constrained, keeping the gradients around (and reducing the amout of allocations) is a good thing for performance. However when we are memory constrained, setting to none gives us more memory to use.

When we are using CPUOffload though, the memory we are keeping is just on the CPU. So there isn't really a GPU memory cost to keeping them around!

```python
optimizer.zero_grad(set_to_none=args.cpu_offload == "off")
```

## Forward Prefetch

TODO

## Not Limiting All Gathers

TODO

## Launch command

We provide a customized launch.sh script here based on the bash command for spawning torchrun on all available nodes:

```bash
cd distributed-training-guide/10-training-llama-405b
bash launch.sh # NOTE: this is non blocking
```

Also note that this launch.sh specifies `HF_HOME` as an environment variable in the tmux session, so if you've not used the default value of `/home/ubuntu/.cache/huggingface`, please update the script!

You can change the hostnames in the `hosts` file in this directory.

## Monitoring

The log files are really useful for monitoring the progress of everything. Here's a bash command for tailing all of them at once:

```bash
cd distributed-training-guide/10-training-llama-405b
find ../logs/ -name \*stderr.log | xargs tail -f
```

Additionally, we have a top like utility script for monitoring the entire cluster at the top level of this directory:

```bash
cd distributed-training-guide/10-training-llama-405b
python ../top-cluster.py hosts
```

## Run statistics

### Memory Usage

### Throughput

Base - 5.1s per iter
nn.Embedding in wrap policy - 4.1s per iter
forward_prefetch=True - ???
limit_all_gathers=False - ???