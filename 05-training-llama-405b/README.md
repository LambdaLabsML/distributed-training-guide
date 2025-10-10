# Training a 405B model

**NOTE: This chapter's code builds off of [chapter 4's FSDP code](../04-fully-sharded-data-parallel/).**

Here we are going to utilize an 8 node cluster (64 H100 GPUs) to train Llama 3.1 405B. **This does not utilize LORA!** We are actually fully training the weights of a 405b model in plain pytorch.

The next few sections go through various changes we have to make to our FSDP code from chapter 4 to make training a 405b model work.

Quick Jump:
- [Use flash attention](#use-flash-attention)
- [Download model weights](#download-model-weights)
- [Loading pretrained weights](#loading-pretrained-weights)
- [Sharding Llama 405B](#sharding-llama-405b)
- [Gradient (aka activation) checkpointing](#gradient-aka-activation-checkpointing)
- [CPU Offload \& fused optimizer kernels](#cpu-offload--fused-optimizer-kernels)
- [NOT de-allocating gradients](#not-de-allocating-gradients)
- [Launch command](#launch-command)
- [Monitoring](#monitoring)
- [Run statistics](#run-statistics)

## Use flash attention

Flash attention is a fused implementation of scaled dot product attention that heavily minimizes memory usage. The whole goal behind it is to query memory as little as possible, and minimize temporary memory used.

Check out the [repo](https://github.com/Dao-AILab/flash-attention) and the [paper](https://arxiv.org/abs/2205.14135) for more information.

This ends up saving us 10s of gb in the forward/backward pass.

Install:

```bash
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation
```

Use it when we initialize our model:

```python
model = AutoModelForCausalLM.from_pretrained(
    ...
    attn_implementation="flash_attention_2",
)
```

## Download model weights

The actual model weights are huge - it contains 191 separate files which are each about 4GB - totally about 764 GB.

There are two options for storing these weights here (and they make a difference!):

1. A shared network drive that all the nodes can access
2. Locally on the main rank 0 node

Node local storage is **much** faster when initializing. For some numbers, while running this script on 8 8xH100 80GB nodes, the shared network drive took 50 minutes to initialize, while the node local storage only took 3 minutes.

There's a download script in this repo for utility, run this on node 0:

```bash
cd distributed-training-guide/05-training-llama-405b
python download.py
```

And run this on the other nodes (to download config & tokenizer):

```bash
cd distributed-training-guide/05-training-llama-405b
python download.py --skip-model
```

NOTE: you will likely have to log into your huggingface account using `hf auth login`.

## Loading pretrained weights

### Load the model into CPU RAM

First we load the full model into CPU RAM on __rank 0__. We will use this later to broadcast all weights to all ranks. Note that we use `from_pretrained()`.

```python
if rank == 0:
    with torch.device("cpu"):
        full_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=dtype
        )
        full_sd = full_model.state_dict()
else:
    full_model = None
    full_sd = {}
dist.barrier()
```

When we actual load the weights, it will take some time AND takes a lot of memory to load. The full size is about 764 GB, so we need to make sure we have enough RAM to store the weights.

### Initialize meta models on all ranks

For the actual models that we will be sharding we will initialize them with the `meta` model. This is for compatibility with the `set_model_state_dict()` api we will use shortly:

```python
with rank0_first(), torch.device("meta"):
    config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
    model = AutoModelForCausalLM.from_config(
        config,
        dtype=dtype,
        attn_implementation="flash_attention_2",
    )
```

### Apply `fully_shard()`

Same thing we did last chapter. One slight difference here is we are setting `reshard_after_forward=False` on the top level model. Since that is the last layer to execute we can use the weights immediately during the backwards pass.

```python
fsdp_config = dict(
    offload_policy=CPUOffloadPolicy() if args.cpu_offload else None,
    mp_policy=MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32),
)
for decoder in model.model.layers:
    fully_shard(decoder, reshard_after_forward=True, **fsdp_config)
fully_shard(model, reshard_after_forward=False, **fsdp_config)

model.to_empty(device="cpu" if args.cpu_offload else device)
dist.barrier()
```

### Broadcast weights from rank 0

[`set_model_state_dict()`](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_model_state_dict) will do the heavy lifting of broadcasting all of our weights (the `broadcast_from_rank0` means rank 0 will broadcast the `full_sd` to all ranks. remember that `full_sd` only exists on rank 0).

```python
set_model_state_dict(
    model=model,
    model_state_dict=full_sd,
    options=StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=args.cpu_offload,
    ),
)
```

Now unfortunately that's not quite all we need because of a little quirk of pytorch: buffers are not included in a model's state_dict()! That means things like RotaryEmbedding layers will not have the correct data because they don't exist in the state dict. So we have to manually do some extra work to broadcast the buffers from rank 0 as well:

```python
if rank == 0:
    for weight, buffer in zip(full_model.buffers(), model.buffers()):
        buffer.copy_(weight)
        dist.broadcast(buffer.to(device), src=0)
else:
    for buffer in model.buffers():
        device_buffer = buffer.to(device)
        dist.broadcast(device_buffer, src=0)
        buffer.copy_(device_buffer.to(buffer.device))
```

And finally just to clean up our CPU RAM usage we can deallocate the `full_model` since we no longer need it:

```python
del full_sd
# convienient way to force deallocation a model
if rank == 0:
    full_model.to(torch.device("meta"))
del full_model
```

## Gradient (aka activation) checkpointing

Another piece of reducing memory usage is gradient checkpointing (first introduced in [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)). Normally when you do the forward pass, you have to keep the input & output in memory until you run the backward pass. This takes up a lot of memory to keep these intermediate tensors around. With gradient checkpointing, we actually **re-run** the forward pass during backwards to regenerate the output. So we are doing more compute but saving a lot of memory.

The method we are using is kind of a hidden method in pytorch, but this is actually exactly what [accelerate uses under the hood](https://github.com/huggingface/accelerate/blob/v0.34.2/src/accelerate/accelerator.py#L1492) so rest assured that it is a "standard" way of doing it:

This piece of code has to go **after** the fully_shard bits!!! I'm not exactly sure of the reason, but it doesn't work before the FSDP initialization.

```python
fully_shard(...)

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

apply_activation_checkpointing(
    model, checkpoint_wrapper_fn=checkpoint_wrapper, auto_wrap_policy=wrap_policy
)
```

## CPU Offload & fused optimizer kernels

Since the model is so large, we pretty much have to enable [CPU offloading](../04-fully-sharded-data-parallel/README.md#cpu-offload) with FSDP. **When using CPUOffload feature of FSDP, the optimizer entirely runs on the CPU**. This is because there is significant cost to transfer data to and from the GPU when doing `optimizer.step()`. At the time of this being written there are open issues on how to overlap the `optimizer.step()` with the next `forward()` call.

By default the optimizers will use non-fused kernel when running on the CPU which will generate a lot of intermediate tensors. By explicitly using the fused kernel we get a lot of speedup, which is especially important since we are running that step on the CPU:

```python
torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
```

If you want to peek through the pytorch code:
1. [_single_tensor_adamw()](https://github.com/pytorch/pytorch/blob/v2.4.1/torch/optim/adamw.py#L322) is the default implementation used
2. [_fused_adamw()](https://github.com/pytorch/pytorch/blob/v2.4.1/torch/optim/adamw.py#L612) is the fused implementation

## torch.compile

This is an easy win for us to gain some throughput:

```python
model = torch.compile(model)
model.loss_function = torch.compile(model.loss_function)
optimizer.step = torch.compile(optimizer.step)
```

## NOT de-allocating gradients

You may have seen this `set_to_none` argument in [optimizer.zero_grad()](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html). According to the docs:

> This will in general have lower memory footprint, and can modestly improve performance.

Basically `set_to_none=True` will **deallocate the gradients** after they are used. In most GPU cases where we want to save a bit of memory, it is a good thing to de-allocate. However in our case we are using CPU offload, which means all of our gradients are already on the CPU! Since we aren't taking up GPU memory, that means we just have to pay for allocating & de-allocating a lot if we do set to none. So if you set `set_to_none=False` you should actually see a slight speed up for our case!

```python
optimizer.zero_grad(set_to_none=not args.cpu_offload)
```

## Launch command

That's pretty much all the changes you need from our base [FSDP code](../04-fully-sharded-data-parallel/). Now let's launch!

We provide a customized [launch.sh](./launch.sh) script here based on the bash command for spawning torchrun on all available nodes:

```bash
cd distributed-training-guide/05-training-llama-405b
bash launch.sh # NOTE: this is non blocking
```

Also note that this launch.sh specifies `HF_HOME` as an environment variable in the tmux session, so if you've not used the default value of `/home/ubuntu/.cache/huggingface`, please update the script!

You can change the hostnames in the [hosts](./hosts) file in this directory.

## Monitoring

We are using torchrun in our [launch.sh](./launch.sh) script, so we will get an output directory per node with a bunch of sub directories with our log files in them. It's a bit of a pain to manually monitor these, so here's a bash command for tailing all of them at once:

```bash
cd distributed-training-guide/05-training-llama-405b
find ../logs/ -name \*stderr.log | xargs tail -f
```

Additionally, we have a top like utility script for monitoring the entire cluster at the top level of this directory:

```bash
cd distributed-training-guide/05-training-llama-405b
python ../top-cluster.py hosts
```

If you notice any of the nprocs go down or the power usage go down then you know that an error has occurred!

To kill all the processes on all the nodes you can just kill the tmux sessions:

```bash
xargs -a hosts -I{} ssh {} tmux kill-session -t torchrun-llama-405b
```

## Run statistics

Training with `--seq-length 4096` and `--batch-size 1` on 64 H100 gpus (8 separate nodes) has the following stats:

- ~30s per iteration (data/forward/backward/update). Breakdown is
  - data: ~2ms
  - forward: ~7s
  - backward: ~19s
  - update: ~4s
- Peak Memory Allocated: 52.9GB
- Peak Memory Reserved: 77.9GB

**NOTE** these numbers were produced with pytorch 2.5.1 and an older version of the guide, so they may be out of date. 64xH100s are hard to find laying around 😄

