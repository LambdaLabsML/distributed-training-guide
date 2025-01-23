# Tensor Parallelism (TP)

So far we've just been using data parallel techniques. You may have heard of other parallelism techniques, and indeed the [Llama 405B paper](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) actually uses 4D parallelism when training the 405B model:

1. Data parallel (FSDP as we've learned)
2. Tensor parallel (**this chapter**)
3. Context parallel (For long context lengths)
4. Pipeline/model parallel

In this chapter we are going to diving into what tensor parallelism is, before we think about combining it with other types.

## Basics: What is tensor parallelism?

TP splits the model weights **AND** computation across multiple GPUs.

FSDP splits the model weights, but it gathers them back for the computation. Splitting the computation across GPUs is the difference.

A result of this is the world size is scaled **down** by your tensor parallel size => the cost of allgathers/allreduces is reduced. This becomes a big factor when your cluster is large, and TP is a very effective way to scale up!

Here are the benefits of this:
1. The peak GPU memory is reduced - now instead of each GPU fully loading up the full weights for each layer, they now only load `1/num_gpus` of the weights.
2. We now have `per GPU memory * num_gpus` as our amount of memory to use for each layer.
3. Less allgather/allreduce cost

Here are the downsides:
1. Global batch size is reduced
2. Increased code complexity

Note that this can only really be applied to certain modules, but most of the modules in an LLM work with it.

## Ensure all GPUs on a node get the same input

Since we are splitting computation across GPUs, that means all the GPUs we are splitting over need to receive the same input. That is why the global batch size is reduced.

First we are going to create our device mesh. A device mesh is just a way to view your devices in an N-dimensional way. So if you have 8 GPUs, you could organize it into a device mesh like `(2, 2, 2)`, or `(2, 4)`, or `(4, 2)` or even things like `(1, 8)`.

The reason this is helpful is because we are going to name these dimensions, much like we do with tensor dimensions. Similar to how we have a batch and sequence dimension, for our device mesh we are going to have a data parallel and tensor parallel dimension.

```python
gpus_on_node = torch.cuda.device_count()
num_nodes = world_size // gpus_on_node
mesh = dist.device_mesh.init_device_mesh(
    "cuda",
    (num_nodes, gpus_on_node),
    mesh_dim_names=("dp", "tp"),
)
```

So if we have 4 GPUs total, and have a `(2, 2)` device mesh, here are the assignments:

| | Data Parallel Group | Tensor Parallel Group |
| --- | --- | --- |
| GPU 0 | 0 | 0 |
| GPU 1 | 0 | 1 |
| GPU 2 | 1 | 0 |
| GPU 3 | 1 | 1 |

This doesn't actually mean anything unless we update the rest of our code to use these device meshes, so let's see how we do that!

The first place is actually our data sampler, and this is how we get all of our GPUs in the tensor parallel group the same input:

```python
sampler=DistributedSampler(
    ...,
    num_replicas=mesh["dp"].size(),
    # NOTE: every GPU on a node will have the same "dp" rank,
    # meaning they will all receive the same input!
    rank=mesh["dp"].get_local_rank(),
)
```

From GPU 0's perspective above, it would have these arguments to DistributedSampler:

| | num_replicas | rank|
| --- | --- | --- |
| GPU 0 | 2 | 0 |
| GPU 1 | 2 | 0 |
| GPU 2 | 2 | 1 |
| GPU 3 | 2 | 1 |

Because our DP dimension is size of 2, and our first table above actually shows the local_rank that we use to pass to DistributedSampler.

## Parallelizing linear & attention modules

First here are some amazing graphics from PyTorch Lightning that show how these parallelization strategies work:

### Colwise Linear

Shards the weight and bias of a Linear layer along dimension 0.

<image src="https://storage.googleapis.com/lightning-avatars/litpages/01hyz8vg94nc6nk7t10rt8jpt1/a2fe38f3-4a73-4b0f-80da-c273d14cadd9.jpeg" width="640px" />

Image Source: [PyTorchLightning](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning#column-wise-parallel)

### Rowwise Linear

Shards the weight of a Linear layer along dimension 1, and replicates the bias layer.

<image src="https://storage.googleapis.com/lightning-avatars/litpages/01hyz8vg94nc6nk7t10rt8jpt1/6b715900-897d-4b3d-a1b6-8ce48f213acf.jpeg" width="640px" />

Image Source: [PyTorchLightning](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning#row-wise-parallel)


### Chaining colwise & rowwise Linears

Clever use of colwise & rowwise together means we can actually chain these techniques together:

<image src="https://storage.googleapis.com/lightning-avatars/litpages/01hyz8vg94nc6nk7t10rt8jpt1/a1bc6e8a-7146-44c6-b6cf-eec124cfbf74.jpeg" width="640px" />

Image Source: [PyTorchLightning](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning#combined-parallel-layers)


### Code to parallelizing the Llama Decoder Layer

The cool thing about this is that the actual matmuls and flash attention that occur on each device will use smaller matrix sizes!

```python
for layer in model.model.layers:
    tp.parallelize_module(
        layer,
        mesh["tp"],
        {
            "self_attn.q_proj": tp.ColwiseParallel(),
            "self_attn.k_proj": tp.ColwiseParallel(),
            "self_attn.v_proj": tp.ColwiseParallel(),
            "self_attn.o_proj": tp.RowwiseParallel(),

            "mlp.gate_proj": tp.ColwiseParallel(),
            "mlp.up_proj": tp.ColwiseParallel(),
            "mlp.down_proj": tp.RowwiseParallel(),
        },
    )
```

### Parallelizing Embedding layer

The embeddings weight get's sharded along dimension 1. Meaning each GPU holds a different slice of the data associated with each token.

When we run this module, it receives a 2d input of size `(batch, seq)`. The output will contain `(batch, seq, model_dim / num_shards)`.

We then need to shard along the sequence dimension, so including the `output_layouts=Shard(1)` means the final output will be `(batch, seq / num_shards, model_dim)`.

```python
tp.parallelize_module(
    model,
    mesh["tp"],
    {"model.embed_tokens": tp.ColwiseParallel(output_layouts=Shard(1))},
)
```

Because of this, we need to pass `position_ids` in forward call. This is very specific to the llama implementation in transformers because it computes sequence length from the _output of `model.embed_tokens`_. Since we are now sharding that layer's output
on the sequence dimension, the llama transformers code will get the wrong seq length. Passing this directly will let us control the sequence length.

```diff
 with timers["data"], torch.no_grad():
     batch = next(batches)
     batch = {k: v.to(device=device) for k, v in batch.items()}
+    batch["position_ids"] = torch.arange(
+        0, args.seq_length, device=device, dtype=torch.long
+    ).unsqueeze(0)
```

### Parallelizing the final linear layer of the model

```python
tp.parallelize_module(
    model,
    mesh["tp"],
    {
        "lm_head": tp.ColwiseParallel(
            output_layouts=Replicate()
        ),
    },
)
```

We have to include `Replicate()` here because by default colwise shards on the last dimension, but we need the output of the network to be replicated across our TP dimension.

## Parallelizing Norm Layers with SequenceParallel

For normalization layers, it works a bit differently. We don't actually shard the layer's weights at all, instead we do store copies of them one very GPU. Instead, we shard the **input** for this on the sequence dimension!

So our computation is split, and we need to do some work to join the results back together for the other modules:

```diff
 for layer in model.model.layers:
     tp.parallelize_module(
         layer,
         mesh["tp"],
         {
+            "input_layernorm": tp.SequenceParallel(),
+            "self_attn": tp.PrepareModuleInput(
+                input_kwarg_layouts={"hidden_states": Shard(dim=1)},
+                desired_input_kwarg_layouts={"hidden_states": Replicate()},
+            ),
             "self_attn.q_proj": tp.ColwiseParallel(),
             "self_attn.k_proj": tp.ColwiseParallel(),
             "self_attn.v_proj": tp.ColwiseParallel(),
-            "self_attn.o_proj": tp.RowwiseParallel(),
+            "self_attn.o_proj": tp.RowwiseParallel(output_layouts=Shard(1)),
+            "post_attention_layernorm": tp.SequenceParallel(),
+            "mlp": tp.PrepareModuleInput(
+                input_layouts=Shard(dim=1),
+                desired_input_layouts=Replicate(),
+            ),
             "mlp.gate_proj": tp.ColwiseParallel(),
             "mlp.up_proj": tp.ColwiseParallel(),
-            "mlp.down_proj": tp.RowwiseParallel(),
+            "mlp.down_proj": tp.RowwiseParallel(output_layouts=Shard(1)),
         },
     )
```

The `PrepareModuleInput` objects transform how the tensors are split up. E.g. for `self_attn` the hidden_states input is sharded along the 1st dimension because of the `SequenceParallel`, but all the `ColwiseParallel` expect input to be replicated.

And here is the diff for our final output from the network:
```diff
 tp.parallelize_module(
     model,
     mesh["tp"],
     {
+        "model.norm": tp.SequenceParallel(),
         "lm_head": tp.ColwiseParallel(
+            input_layouts=Shard(1),
             output_layouts=Replicate(),
         ),
     },
 )
```

## Parallelizing Loss computation

There's an additional api for parallelizing the loss computation (only works for Cross Entropy at the moment of writing) across the **class** dimension. We first need to use this context manager around our loss computation:

```python
with tp.loss_parallel(), timers["forward"]:
    outputs = model(**batch)

with tp.loss_parallel(), timers["backward"]:
    outputs.loss.backward()
```

Then we need to update the output of our `lm_head` for this also, because loss_parallel requires different sharding format and DTensor:

```diff
 tp.parallelize_module(
     model,
     mesh["tp"],
     {
         "model.norm": tp.SequenceParallel(),
         "lm_head": tp.ColwiseParallel(
             input_layouts=Shard(1),
-            output_layouts=Replicate(),
+            output_layouts=Shard(-1),
+            use_local_output=False,
         ),
     },
 )
```

`use_local_output=False` tells pytorch to return a `DTensor` from the operation, instead of a normal `Tensor`.

## Computing throughput with our new world size

Because each of our GPUs is now no longer the unit, we just need to update our throughput calculation to use our device mesh:

```diff
 if state["global_step"] % args.log_freq == 0:
-    tok_per_step = world_size * args.batch_size * args.seq_length
+    tok_per_step = mesh["dp"].size() * args.batch_size * args.seq_length
     ms_per_step = sum(t.avg_elapsed_ms() for t in timers.values())
```

## Results

Here are some results from launching training for llama 8B on a single node of 8x H100s:

Command:
```bash
HF_HOME=/home/ubuntu/.cache/huggingface OMP_NUM_THREADS=26 torchrun --standalone --nproc-per-node gpu train_llm.py --experiment-name tp-llama-8b --dataset-name tatsu-lab/alpaca --model-name meta-llama/Llama-3.1-8B --log-freq 10 --batch-size 16 --seq-length 1024 --num-epochs 1
```

<img src="https://github.com/user-attachments/assets/4bafada2-beea-4e37-a341-62d7f4639014" width="480px" />

<img src="https://github.com/user-attachments/assets/c6e67666-db7c-4b67-bae4-98480382557f" width="480px" />

## Useful References

For completeness here are the relevant docs/guides from pytorch on how to achieve this:
- [TP API docs](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#tensor-parallelism-torch-distributed-tensor-parallel)
- [2d Parallelism Tutorial](https://pytorch.org/tutorials/intermediate/TP_tutorial.html#large-scale-transformer-model-training-with-tensor-parallel-tp)
- [Device Mesh tutorial](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)
- [PyTorch Lightning TP Tutorial](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning)

## Pytorch API Reference

Here we are going to give a brief explanation of how the api we are going to be using works.

- [tp.RowwiseParallel()](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.RowwiseParallel) shards the module's weights in a row wise fashion.
    - Inputs by default are sharded on last dimension
    - Outputs by default are replicated on all workers
- [tp.ColwiseParallel()](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.ColwiseParallel) shards the module's weights in a col wise fashion.
    - Inputs by default are replicated on all workers
    - Outputs by default are sharded on last dimension
- [tp.SequenceParallel()](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.SequenceParallel) shards the input/output across dimension 1. Module weights are NOT sharded.
- [tp.PrepareModuleInput()](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleInput) let's you change the sharding configuration of input tensors
- `torch.distributed._tensor.Shard(dim=X)` indicates a tensor should be sharded along dimension X
- `torch.distributed._tensor.Replicate()` indicates a tensor should be replicated among all workers.

How all of these things interact is actually very subtle and complex, which is why this guide is useful!

You can also change most of the default behavior with arguments to these classes. For example, you can change RowwiseParallel to assume the input is replicated instead of sharded.
