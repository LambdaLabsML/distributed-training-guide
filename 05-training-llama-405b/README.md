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
- [Other notes on settings that didn't affect throughput](#other-notes-on-settings-that-didnt-affect-throughput)

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

NOTE: you will likely have to log into your huggingface account using `huggingface-cli login`.

## Loading pretrained weights

When we actual load the weights, it will take some time AND takes a lot of memory to load. Again the full size is about 764 GB, so we need to make sure we have enough RAM to store the weights.

There's three parts to this:

1. Loading the weights into RAM only on `rank==0`
2. Using the [meta](../04-fully-sharded-data-parallel/README.md#initialization-after-sharding---the-meta-device) device on `rank>0`
3. Using `from_config` instead of `from_pretrained` on `rank>0` so we don't need to download the weights on all the nodes.
   1. Note that if you have the weights on a shared network drive, you can just use `from_pretrained` instead.
4. Enabling [sync_module_states](../04-fully-sharded-data-parallel/README.md#sync_module_states) in FSDP constructor

You might think of using the `device_map` feature of `transformers` - e.g. `device_map="auto"` tries to smartly fill up memory. However if you try this approach you'll end up with out of memory errors when FSDP tries to start sending memory to the GPU.

Here's our code snippet for doing this:

```python
if rank == 0:
    with torch.device("cpu"):
        model = AutoModelForCausalLM.from_pretrained(...)
else:
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
```

Then later, sync_module_states in [FSDP constructor](../04-fully-sharded-data-parallel/README.md#the-fsdp-constructor) will make sure the weights are broadcasted from rank 0 to the other ranks.

## Sharding Llama 405B

Determining what layers you should shard is complex. If you are using `transformers`, they include a private attribute on classes called [_no_split_modules](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/models/llama/modeling_llama.py#L784) that will contain classes that you should not shard anything under them. E.g. for Llama this attribute just contains `LlamaDecoderLayer`. So that is what we will wrap! During testing I also found that sharding the `nn.Embedding` layer at the beginning of the network improved throughput and reduced memory usage.

We can use the [transformer_auto_wrap_policy()](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/wrap.py#L307C5-L307C33) to target the specific classes for those layers, and pass that as our [auto_wrap_policy in the FSDP constructor](../04-fully-sharded-data-parallel/README.md#what-layers-to-shard---the-auto_wrap_policy):

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer, nn.Embedding},
)
FSDP(..., auto_wrap_policy=wrap_policy)
```

Please consult [our explanation on the FSDP constructor](../04-fully-sharded-data-parallel/README.md#the-fsdp-constructor) for more info.

As a reminder - this will cause FSDP to gather all the parameters for each DecoderLayer (which includes Attention, Linear, and various norm modules), and shard them across the world. At the start of forward/backward pass FSDP will issue an all-gather so all the nodes have the full weights in memory, and at the end of the DecoderLayer forward/backward, it will free up the full weights again.

So where you apply FSDP determines where the all-gather happens!

## Gradient (aka activation) checkpointing

Another piece of reducing memory usage is gradient checkpointing (first introduced in [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)). Normally when you do the forward pass, you have to keep the input & output in memory until you run the backward pass. This takes up a lot of memory to keep these intermediate tensors around. With gradient checkpointing, we actually **re-run** the forward pass during backwards to regenerate the output. So we are doing more compute but saving a lot of memory.

The method we are using is kind of a hidden method in pytorch, but this is actually exactly what [accelerate uses under the hood](https://github.com/huggingface/accelerate/blob/v0.34.2/src/accelerate/accelerator.py#L1492) so rest assured that it is a "standard" way of doing it:

This piece of code has to go **after** the FSDP constructor!!! I'm not exactly sure of the reason, but it doesn't work before the FSDP initialization.

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

model = FSDP(...)

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

## NOT de-allocating gradients

You may have seen this `set_to_none` argument in [optimizer.zero_grad()](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html). According to the docs:

> This will in general have lower memory footprint, and can modestly improve performance.

Basically `set_to_none=True` will **deallocate the gradients** after they are used. In most GPU cases where we want to save a bit of memory, it is a good thing to de-allocate. However in our case we are using CPU offload, which means all of our gradients are already on the CPU! Since we aren't taking up GPU memory, that means we just have to pay for allocating & de-allocating a lot if we do set to none. So if you set `set_to_none=False` you should actually see a slight speed up for our case!

```python
optimizer.zero_grad(set_to_none=args.cpu_offload == "off")
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

Noting that reserved memory has to do with pytorch allocation caching.

## Other notes on settings that didn't affect throughput

- Allowing tf32 had no impact on throughput (`torch.backends.cudnn.allow_tf32` and `torch.backends.cuda.matmul.allow_tf32`) 
- Enabling benchmarking had no impact on throughput (`torch.backends.cudnn.benchmark = True`)
- Using CuDNN sdpa was slower (`attn_implementation="sdpa"` and `torch.backends.cuda.enable_cudnn_sdp(True)`)
- torch.compile had no impact (`use_orig_params=True` and `torch.compile` after FSDP constructor)
- Very minimal testing of NCCL environment variables either made things worse or had no impact (https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- `PYTORCH_NO_CUDA_MEMORY_CACHING=1` made enough memory available that `--batch-size 2` or higher sequence lengths were possible, but it was much much slower.
  - It's possible that some well placed calls to `torch.cuda.empty_cache()` could achieve this without the throughput loss.
- Only `FULL_SHARD` works. Others fail silently.
