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

## Auto Wrap - LlamaDecoderLayer

TODO

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

## Monitoring rank 0 loading progress

Initializing the model takes a **loooooong** time. On 8 8xH100 nodes (64 gpus), it took me ~50 minutes.

When using local node memory instead shared network memory it takes: TODO minutes.

To monitor the progress of this there's two things you can do:

1. Watch the Host memory on `nvtop` for GPU0.
2. Watch the rank 0 process with `py-spy top --pid <rank 0 pid>`

I haven't found a way in python to have a progress bar, because the functions that take the most time
are deep in pytorch code.

## Monitoring Logs

Tailing all torchrun log files at once:

```bash
find ../logs/ -name \*.log | xargs tail -f
```

Finding rank 0 log file:

```bash
find ../logs/ -name \*.log | xargs grep "rank=0"
```

## Memory Usage

