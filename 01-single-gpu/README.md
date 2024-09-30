# Single GPU

This is the "standard" single gpu training script. It doesn't do anything with distributed, and aims to be as simple as possible.

## Command

```bash
cd distributed-training-guide/01-single-gpu
python train_llm.py \
    --experiment-name gpt2-alpaca-single-gpu-$(date +%Y-%m-%dT%H-%M-%S) \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2 \
    --batch-size 64
```

## Notable Features

This is a basic train from scratch script. Here are some quick facts:

1. We are training a BF16 causal language model (think GPT) using `transformers`

```python
device = torch.device("cuda")
dtype = torch.bfloat16

config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
model = AutoModelForCausalLM.from_config(config).to(dtype=dtype, device=device)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
```

2. We save checkpoints into `args.save_dir/args.experiment_name` - `--experiment-name is a **unique** run identifier

```python
exp_dir: Path = Path(args.save_dir) / args.experiment_name
```

3. If `args.save_dir/args.experiment_name` already exists, we attempt to resume

```python
state = {
    "epoch": 0,
    "global_step": 0,
    "epoch_step": 0,
    "running_loss": 0,
}
resumed = False
if (exp_dir / "state.json").exists():
    model.load_state_dict(_load_to_device(exp_dir / "model.pt"))
    optimizer.load_state_dict(_load_to_device(exp_dir / "optimizer.pt"))
    lr_scheduler.load_state_dict(_load_to_device(exp_dir / "lr_scheduler.pt"))
    with open(exp_dir / "state.json") as fp:
        state = json.load(fp)
    resumed = True
```

4. We resume the run in wandb if we loaded a checkpoint (& also ensure that our unique experiment ID is used for the wandb run id)

```diff
wandb.init(
    project="distributed-training-guide",
    dir=exp_dir,
    name=args.experiment_name,
    id=args.experiment_name,
+    resume="must" if resumed else None,
    save_code=True,
    config={
        "args": vars(args),
        "embedding_size": len(tokenizer),
        "training_data_size": len(train_data),
        "num_batches": len(dataloader),
    },
)
```

5. We are timing the various parts of the inner training loop (to help us understand distributed efficiency)

```diff
+ with timers["data"], torch.no_grad():
    batch = {k: v.to(device=device) for k, v in batch.items()}

+ with timers["forward"]:
    outputs = model(**batch)

+ with timers["backward"]:
    optimizer.zero_grad()
    outputs.loss.backward()

+ with timers["update"]:
    optimizer.step()
    lr_scheduler.step()
```

6. We log various things to wandb (to help us visualize multiple workers later)

```python
wandb.log(
    {
        "lr": lr_scheduler.get_last_lr()[0],
        "running_loss": state["running_loss"] / args.log_freq,
        "epoch": state["epoch"],
        "epoch_progress": state["epoch_step"] / len(dataloader),
        "num_batches_remaining": len(dataloader) - i_step,
        "time/total": sum(t.avg_elapsed_ms() for t in timers.values()),
        **{
            f"time/{k}": timer.avg_elapsed_ms()
            for k, timer in timers.items()
        },
    },
    step=state["global_step"],
)
```