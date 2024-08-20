import argparse
from itertools import chain
import json
import multiprocessing
import random
import time
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

import deepspeed
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

_LOGGER = logging.getLogger(__name__)


@record
def main():
    logging.basicConfig(level=logging.INFO)

    parser = _get_parser()
    args = parser.parse_args()

    _LOGGER.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    numpy.random.seed(args.seed)
    random.seed(args.seed)

    deepspeed.init_distributed(
        dist_backend="nccl" if dist.is_nccl_available() else "mpi"
    )

    rank = dist.get_rank()
    if args.local_rank is not None:
        local_rank = args.local_rank
    else:
        local_rank = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()

    _LOGGER.info(f"local rank={local_rank} rank={rank} world size={world_size}")

    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    torch.cuda.set_device(device)

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    with deepspeed.zero.Init():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(
            dtype=dtype, device=device
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # NOTE: since this can download data, make sure to do the main process first
    # NOTE: This assumes that the data is on a **shared** network drive, accessible to all processes
    if rank == 0:
        train_data = _load_and_preprocess_data(args, tokenizer, config)
    dist.barrier()
    if rank > 0:
        train_data = _load_and_preprocess_data(args, tokenizer, config)
    _LOGGER.info(f"[{rank}] {len(train_data)} training samples")

    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        # NOTE: this sampler will split dataset evenly across workers
        sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
    )
    _LOGGER.info(f"[{rank}] {len(dataloader)} batches per epoch")

    model_engine: deepspeed.DeepSpeedEngine
    model_engine, _, _, lr_scheduler = deepspeed.initialize(
        args,
        model=model,
        model_parameters=(p for p in model.parameters() if p.requires_grad),
    )

    exp_dir: Path = Path(args.save_dir) / args.experiment_name

    # attempt resume
    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0,
    }
    resumed = False
    if (exp_dir / "pytorch_model.bin").exists():
        load_path, state = model_engine.load_checkpoint(exp_dir)
        resumed = load_path is not None
    _LOGGER.info(f"[{rank}] Resumed={resumed} | {state}")

    dist.barrier()
    if rank == 0:
        # NOTE: assuming directory is shared across all nodes, that's why we do rank instead of local_rank
        _LOGGER.info(f"Creating experiment root directory")
        exp_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    (exp_dir / f"rank-{rank}").mkdir(parents=True, exist_ok=True)
    _LOGGER.info(f"[{rank}] Worker saving to {exp_dir / f'rank-{rank}'}")

    wandb.init(
        project="distributed-training-guide",
        dir=exp_dir / f"rank-{rank}",
        group=args.experiment_name,
        name=f"rank-{rank}",
        id=f"{args.experiment_name}-{rank}",
        resume="must" if resumed else None,
        save_code=True,
        config={
            "args": vars(args),
            "embedding_size": len(tokenizer),
            "training_data_size": len(train_data),
            "num_batches": len(dataloader),
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
        },
    )

    timers = {k: LocalTimer(device) for k in ["data", "forward", "backward", "update"]}

    for state["epoch"] in range(state["epoch"], args.num_epochs):
        _LOGGER.info(
            f"[{rank}] Begin epoch {state['epoch']} at step {state['epoch_step']}"
        )

        progress_bar = tqdm.tqdm(range(len(dataloader)), disable=rank > 0)
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])

        batches = iter(dataloader)

        for i_step in range(len(dataloader)):
            with timers["data"], torch.no_grad():
                batch = next(batches)
                batch = {k: v.to(device=device) for k, v in batch.items()}

            if i_step < state["epoch_step"]:
                # NOTE: for resuming
                continue

            with timers["forward"]:
                outputs = model_engine(**batch)

            with timers["backward"]:
                model_engine.backward(outputs.loss)

            with timers["update"]:
                model_engine.step()

            state["global_step"] += 1
            state["epoch_step"] += 1
            state["running_loss"] += outputs.loss.item()
            progress_bar.update(1)

            if state["global_step"] % args.log_freq == 0:
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
                state["running_loss"] = 0
                for t in timers.values():
                    t.reset()

            if state["global_step"] % args.ckpt_freq == 0:
                model_engine.save_checkpoint(exp_dir, client_state=state)
                dist.barrier()

        state["epoch_step"] = 0


def _load_and_preprocess_data(args, tokenizer, config):
    data = datasets.load_dataset(
        args.dataset_name, trust_remote_code=True, cache_dir=args.dataset_cache_root
    )

    column_names = data["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = data.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
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
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return lm_datasets["train"]


class LocalTimer:
    def __init__(self, device: torch.device):
        if device.type == "cpu":
            self.synchronize = lambda: torch.cpu.synchronize(device=device)
        elif device.type == "cuda":
            self.synchronize = lambda: torch.cuda.synchronize(device=device)
        self.measurements = []
        self.start_time = None

    def __enter__(self):
        self.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if traceback is None:
            self.synchronize()
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
    parser.add_argument("--save-dir", default="../outputs")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--log-freq", default=100, type=int)
    parser.add_argument("--ckpt-freq", default=500, type=int)
    parser.add_argument("--dataset-cache-root", default="../.cache")
    parser.add_argument("--local_rank", type=int, default=None)
    deepspeed.add_config_arguments(parser)
    return parser


if __name__ == "__main__":
    main()
