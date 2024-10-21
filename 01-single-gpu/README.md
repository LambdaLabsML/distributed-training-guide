# Single GPU

This is the "standard" single gpu training script. It doesn't do anything with distributed, and aims to be as simple as possible.

The rest of this guide uses this code as the basis, so this chapter explains all the different parts of the code and why we do them.

## Command

```bash
cd distributed-training-guide/01-single-gpu
python train_llm.py \
    --experiment-name gpt2-alpaca-single-gpu-$(date +%Y-%m-%dT%H-%M-%S) \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2
```

## Code explanation

This explanation goes roughly in code order, starting from the top.

### Argument parsing

Our training script is a CLI (command line interface) program. That means you run it from a terminal. We have a variety of arguments we'd like the user (you) to be able to change using the CLI. So this is a very standar python way to enable that:

```python
def main():
    parser = _get_parser()
    args = parser.parse_args()


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default=None, required=True)
    parser.add_argument("--dataset-name", default=None, required=True)
    parser.add_argument("--model-name", default=None, required=True)
    parser.add_argument("--save-dir", default="../outputs")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--log-freq", default=100, type=int)
    parser.add_argument("--ckpt-freq", default=500, type=int)
    return parser


if __name__ == "__main__":
    main()
```

### Setting up logging

For this guide, we just use the built in `logging` package for python. This will output everything to stdout/stderr, and we use command line tools to redirect this output to files for later.

```python
LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    format=f"[%(asctime)s] %(levelname)s:%(message)s",
    level=logging.INFO,
)

LOGGER.info(os.environ)
LOGGER.info(args)
```

It's useful to be able to see what the environment variables & CLI args we are running the program with (especially with multiple nodes involved later). So we log those first.

### pytorch setup

As we are using pytorch there are a couple of useful things to do before we initialize anything

```python
device = torch.device("cuda")
dtype = torch.bfloat16
torch.manual_seed(args.seed)
```

Here we are saying that the device we will be using for the rest of the script is a GPU (specifically a CUDA device), and that we are going to be training with bfloat16 (aka bf16) which is a 16 bit floating point number (float is 32 bit, and double is 64 bits).

### Initializing the model

We are training a BF16 causal language model (think GPT) using `transformers`

```python
config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
with device:
    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
```

Note that the `with device:` will construct all tensors on our device immediately, so we don't have to allocate on the CPU and then transfer to the device.

### Initializing our dataset

We are using `datasets` to load and preprocess our dataset. The processing code used in this guide was sourced from https://github.com/huggingface/transformers/blob/v4.45.1/examples/pytorch/language-modeling/run_clm_no_trainer.py.

Encourage readers to check out datasets if they want more information.

### Data Loading, LR Schedule, Optimizer

The next section of code is fairly standard pytorch. We are using a DataLoader to iterate our dataset, the AdamW optimizer, and a Cosine Annealing LR schedule.

```python
dataloader = DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=default_data_collator,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=1000, eta_min=args.lr * 1e-2
)
```

### Outputs & Resuming

We save checkpoints into `args.save_dir/args.experiment_name` - `--experiment-name is a **unique** run identifier

```python
exp_dir: Path = Path(args.save_dir) / args.experiment_name
```

If `args.save_dir/args.experiment_name/state.json` already exists, we attempt to resume. This means if a checkpoint already exists for our experiment_name, then we interpret this as a resumed run.

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

### Experiment tracking with Weights & Biases (wandb)

We resume the run in [wandb](https://wandb.ai/) if we loaded a checkpoint (& also ensure that our unique experiment ID is used for the wandb run id).

We include a couple of useful initialization flags here as well, so wandb will save our code, and include some hyperparameters we specified on the CLI.

When we resume a run, we tell wandb that we "must" initialize in resume mode.

```python
wandb.init(
    project="distributed-training-guide",
    dir=exp_dir,
    name=args.experiment_name,
    id=args.experiment_name,
    resume="must" if resumed else None,
    save_code=True,
    config={
        "args": vars(args),
        "training_data_size": len(train_data),
        "num_batches": len(dataloader),
    },
)
```

### Iterating our batches

We do this in a non-standard way so we can time various parts of the training loop. Normally, we wouldn't be able to time the actual construction of the batch, but by manually pulling the next batch using `next()`, we can time it:

```python
batches = iter(dataloader)

for i_step in range(len(dataloader)):
    # Here we measure the time it takes to generate a batch and move it to the GPU
    with timers["data"], torch.no_grad():
        batch = next(batches)
        batch = {k: v.to(device=device) for k, v in batch.items()}
```

### Forward/backward/update

This is standard pytorch code, with the addition of timing so we can benchmark:

```python
with timers["forward"]:
    outputs = model(**batch)

with timers["backward"]:
    # NOTE: set_to_none=True will de-allocate the gradients, saving us some memory.
    optimizer.zero_grad(set_to_none=True)
    outputs.loss.backward()

with timers["update"]:
    optimizer.step()
    lr_scheduler.step()
```

### Logging to wandb (& stdout)

The next blocks of code involve logging various tidbits about how our training is going:

We do this based on the `--log-freq` argument, e.g. if we do `--log-freq 100` we will log this data every 100 steps.

Note that we both log to our LOGGER, and also wandb.

```python
if state["global_step"] % args.log_freq == 0:
    info = {
        "global_step": state["global_step"],
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
    }

    LOGGER.info(info)
    wandb.log(info, step=state["global_step"])

    state["running_loss"] = 0
    for t in timers.values():
        t.reset()
```

### Checkpoints

The final block of code is our checkpointing logic, here just using `torch.save`.

Note that we are saving the optimizer and LR scheduler in addition to the model!

```python
if state["global_step"] % args.ckpt_freq == 0:
    LOGGER.info("Saving checkpoint.")
    torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
    torch.save(model.state_dict(), exp_dir / "model.pt")
    torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
    with open(exp_dir / "state.json", "w") as fp:
        json.dump(state, fp)
```
