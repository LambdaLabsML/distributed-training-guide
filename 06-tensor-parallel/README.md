# Tensor Parallelism (TP)

So far we've just been using data parallel techniques. You may have heard of other parallelism techniques, and indeed the [Llama 405B paper](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) actually uses 4D parallelism when training the 405B model:

1. Data parallel (FSDP as we've learned)
2. Tensor parallel (this chapter)
3. Context parallel (For long context lengths)
4. Pipeline/model parallel

In this chapter we are going to diving into what tensor parallelism is, before we think about combining it with other types.

## Basics: What is tensor parallelism?

TP splits the model weights & computation across multiple GPUs.

Sharded data parallel does split the model weights, but it gathers them back for the computation. The computation bit being split across GPUs is the difference.

We will be treating an entire node of GPUs as the execution unit:
1. The model is sharded across the GPUs on the node.
2. Every *GPU* on the node receives **the same input**

It reduces the world size by however many GPUs are in your node, meaning the cost of allgathers/allreduces is reduced. This becomes a big factor when your cluster is large.

It's a very effective way to scale up!

Here are the benefits of this:
1. The peak GPU memory is reduced - now instead of each GPU fully loading up the full weights for each layer, they now only load `1/num_gpus` of the weights.
2. We now have `per GPU memory * num_gpus` as our amount of memory to use for each layer.
3. Less allgather/allreduce cost

Here are the downsides:
1. The global batch size is reduced
2. BUT we can now hopefully increase the local batch size per node, since we are reducing peak GPU memory.

Note that this can only really be applied to certain modules, but most of the modules in an LLM work with it.

## First: ensure all GPUs on a node get the same input

First we are going to create our device mesh. This doesn't do anything on its own, but we will be passing this object to other APIs.

```python
gpus_on_node = torch.cuda.device_count()
num_nodes = world_size // gpus_on_node
mesh = dist.device_mesh.init_device_mesh(
    "cuda",
    (num_nodes, gpus_on_node),
    mesh_dim_names=("dp", "tp"),
)
```

Now with a small update to our sampler we can pass all of our TP GPUs the same input:

```python
sampler=DistributedSampler(
    ...,
    num_replicas=mesh["dp"].size(),
    # NOTE: every GPU on a node will have the same "dp" rank,
    # meaning they will all receive the same input!
    rank=mesh["dp"].get_local_rank(),
)
```

Our new world size is size of the data parallel mesh, and our new rank is our processes rank in the data parallel mesh. Note that all of our processes on the same node will have the same data parallel rank, and this is what we want!

## Parallelizing linear & attention modules

First here are some amazing graphics from PyTorch Lightning that show how these parallelization strategies work:

### Colwise Linear

Shards the weight and bias of a Linear layer along dimension 0.

![image](https://storage.googleapis.com/lightning-avatars/litpages/01hyz8vg94nc6nk7t10rt8jpt1/a2fe38f3-4a73-4b0f-80da-c273d14cadd9.jpeg)

Image Source: [PyTorchLightning](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning#column-wise-parallel)

### Rowwise Linear

Shards the weight of a Linear layer along dimension 1, and replicates the bias layer.

![image](https://storage.googleapis.com/lightning-avatars/litpages/01hyz8vg94nc6nk7t10rt8jpt1/6b715900-897d-4b3d-a1b6-8ce48f213acf.jpeg)

Image Source: [PyTorchLightning](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning#row-wise-parallel)


### Chaining colwise & rowwise Linears

Clever use of colwise & rowwise together means we can actually chain these techniques together:

![image](https://storage.googleapis.com/lightning-avatars/litpages/01hyz8vg94nc6nk7t10rt8jpt1/a1bc6e8a-7146-44c6-b6cf-eec124cfbf74.jpeg)

Image Source: [PyTorchLightning](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning#combined-parallel-layers)


### Code to parallelizing the Llama Decoder Layer

```
for layer in model.model.layers:
    # Have the adjust these values since we are sharding the linear layers
    layer.self_attn.num_heads //= mesh["tp"].size()
    layer.self_attn.num_key_value_heads //= mesh["tp"].size()

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

```python
tp.parallelize_module(
    model,
    mesh["tp"],
    {"model.embed_tokens": tp.ColwiseParallel()},
)
```

### Parallelizing the final linear layer of the model

```
tp.parallelize_module(
    model,
    mesh["tp"],
    {
        "lm_head": tp.ColwiseParallel(
            output_layouts=Replicate(),
            use_local_output=True,
        ),
    },
)
```

## Parallelizing Norm Layers with SequenceParallel

For normalization layers, it works a bit differently. We don't actually shard the layer's weights at all, instead we do store copies of them one very GPU. Instead, we shard the **input** for this on the sequence dimension!

TODO

## Transformers implementation extra changes

This is very specific to the llama implementation in transformers, but we need to add some additionally input to our model to make this all work properly (otherwise the llama implementation will get some of the sequence dimensions incorrect):

```diff
 with timers["data"], torch.no_grad():
     batch = next(batches)
     batch = {k: v.to(device=device) for k, v in batch.items()}
+    batch["position_ids"] = torch.arange(
+        0, args.seq_length, device=device, dtype=torch.long
+    ).unsqueeze(0)
```

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

![W B Chart 1_10_2025, 11_07_19 AM](https://github.com/user-attachments/assets/4bafada2-beea-4e37-a341-62d7f4639014)

![W B Chart 1_10_2025, 11_07_04 AM](https://github.com/user-attachments/assets/c6e67666-db7c-4b67-bae4-98480382557f)

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
