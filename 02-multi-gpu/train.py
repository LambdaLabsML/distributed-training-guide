import argparse
from itertools import chain
import json
import logging
import random
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
import numpy
import wandb
import tqdm
import datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)


def main():
    parser = _get_parser()
    args = parser.parse_args()

    experiment_dir = os.path.join(args.save_dir, args.experiment_name)
    optimizer_path = os.path.join(experiment_dir, "optimizer.pt")
    model_path = os.path.join(experiment_dir, "model.pt")
    state_path = os.path.join(experiment_dir, "state.json")
    lr_scheduler_path = os.path.join(experiment_dir, "lr_scheduler.pt")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    numpy.random.seed(args.seed)
    random.seed(args.seed)

    dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "mpi")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.disabled = rank > 0
    logger.info(f"{args}")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(
        dtype=dtype, device=device
    )
    model = DDP(model, device_ids=[rank])

    train_data = _load_and_preprocess_data(args, config)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=args.lr * 1e-2
    )

    # attempt resume
    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0,
    }
    resumed = False
    if os.path.exists(experiment_dir):
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location=device)
        )
        optimizer.load_state_dict(
            torch.load(optimizer_path, weights_only=True, map_location=device)
        )
        lr_scheduler.load_state_dict(
            torch.load(lr_scheduler_path, weights_only=True, map_location=device)
        )
        with open(state_path) as fp:
            state = json.load(fp)
        logger.info(f"Resumed from {experiment_dir} | {state}")
        resumed = True

    wandb.init(
        dir=experiment_dir,
        name=args.experiment_name,
        id=args.experiment_name,
        resume="must" if resumed else None,
        save_code=True,
        config=vars(args),
    )

    timers = {k: LocalTimer(device) for k in ["data", "forward", "backward", "step"]}

    for state["epoch"] in range(state["epoch"], args.num_epochs):
        progress_bar = tqdm.tqdm(range(len(train_loader)), disable=rank > 0)
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])
        for i_step, batch in enumerate(train_loader):
            if i_step < state["epoch_step"]:
                # NOTE: for resuming
                continue

            with timers["data"], torch.no_grad():
                batch = {k: v.to(device=device) for k, v in batch.items()}

            with timers["forward"]:
                outputs = model(**batch)

            with timers["backward"]:
                optimizer.zero_grad()
                outputs.loss.backward()

            with timers["step"]:
                optimizer.step()
                lr_scheduler.step()

            state["global_step"] += 1
            state["epoch_step"] += 1
            state["running_loss"] += outputs.loss.item()
            progress_bar.update(1)

            if state["global_step"] % args.log_freq == 0:
                wandb.log(
                    {
                        f"lr/{rank}": lr_scheduler.get_last_lr()[0],
                        f"running_loss/{rank}": state["running_loss"] / args.log_freq,
                        f"epoch/{rank}": state["epoch"],
                        f"time/total/{rank}": sum(
                            t.avg_elapsed_ms() for t in timers.values()
                        ),
                        **{
                            f"time/{k}/{rank}": timer.avg_elapsed_ms()
                            for k, timer in timers.items()
                        },
                    },
                    step=state["global_step"],
                )
                state["running_loss"] = 0
                for t in timers.values():
                    t.reset()

            if state["global_step"] % args.ckpt_freq == 0:
                logger.info(f"{state}")

                if rank == 0:
                    os.makedirs(experiment_dir, exist_ok=True)
                    torch.save(optimizer.state_dict(), optimizer_path)
                    torch.save(model.state_dict(), model_path)
                    torch.save(lr_scheduler.state_dict(), lr_scheduler_path)
                    with open(state_path, "w") as fp:
                        json.dump(state, fp)
                dist.barrier()

        state["epoch_step"] = 0


def _load_and_preprocess_data(args, config):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    data = datasets.load_dataset(args.dataset_name, trust_remote_code=True)

    column_names = data["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = data.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    block_size = tokenizer.model_max_length
    if block_size > config.max_position_embeddings:
        block_size = min(1024, config.max_position_embeddings)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return lm_datasets["train"]


class LocalTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.measurements = []
        self.start_time = None

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(device=self.device)
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if traceback is None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(device=self.device)
            end_time = time.time()
            self.measurements.append(end_time - self.start_time)
        self.start_time = None

    def avg_elapsed_ms(self):
        return 1000 * (sum(self.measurements) / len(self.measurements))

    def reset(self):
        self.measurements = []
        self.start_time = None


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default=None, required=True)
    parser.add_argument("--dataset-name", default=None, required=True)
    parser.add_argument("--model-name", default=None, required=True)
    parser.add_argument("--save-dir", default=".")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--log-freq", default=100, type=int)
    parser.add_argument("--ckpt-freq", default=500, type=int)
    return parser


if __name__ == "__main__":
    try:
        main()
    finally:
        wandb.finish()
        dist.destroy_process_group()
