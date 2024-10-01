import argparse
from contextlib import contextmanager
import functools
from itertools import chain
import json
import multiprocessing
import os
import time
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel,
    BackwardPrefetch,
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.state_dict_loader import load
from torch.distributed.checkpoint.state_dict_saver import save


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
    parser = _get_parser()
    args = parser.parse_args()

    dist.init_process_group()

    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()

    logging.basicConfig(
        format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )

    _LOGGER.info(os.environ)
    _LOGGER.info(args)
    _LOGGER.info(f"local_rank={local_rank} rank={rank} world size={world_size}")

    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    torch.cuda.set_device(device)

    torch.manual_seed(args.seed)

    with rank0_first():
        config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # NOTE: meta device will not allocate any memory
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    _LOGGER.info(
        f"Before FSDP: {torch.cuda.memory_stats(device)['allocated_bytes.all.current'] * 1e-9}gb allocated"
    )

    def safe_param_init_fn(module: torch.nn.Module):
        """
        For use in FSDP constructor. This is identical to default behavior of FSDP when dealing with meta device,
        except pytorch code doesn't check for existence of `reset_parameters()` before calling it. Some modules
        don't have this implemented, so this is our "fix" for it.
        """
        # NOTE: according to FSDP.__init__.param_init_fn documnetaiton, we should set recurse=False
        module.to_empty(device=device, recurse=False)
        # NOTE: Since we are training from scratch here, we just reset the parameters,
        #       otherwise we may want to load in weights directly here, or load
        #       parameters on rank 0 and use sync_module_states=True in FSDP constructor.
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=int(args.numel_to_wrap)
    )
    model = FullyShardedDataParallel(
        model,
        device_id=local_rank,
        param_init_fn=safe_param_init_fn,
        sync_module_states=True,
        # NOTE: FULL_SHARD is equivalent to deepspeed ZeRO stage 3
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=args.cpu_offload == "on"),
        backward_prefetch=getattr(BackwardPrefetch, args.bwd_prefetch, default=None),
    )

    _LOGGER.info(
        f"After FSDP: {torch.cuda.memory_stats(device)['allocated_bytes.all.current'] * 1e-9}gb allocated"
    )

    # NOTE: since this can download data, make sure to do the main process first
    # NOTE: This assumes that the data is on a **shared** network drive, accessible to all processes
    with rank0_first():
        train_data = _load_and_preprocess_data(args, tokenizer, config)
    _LOGGER.info(f"{len(train_data)} training samples")

    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        # NOTE: this sampler will split dataset evenly across workers
        sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
    )
    _LOGGER.info(f"{len(dataloader)} batches per epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=args.lr * 1e-2
    )

    exp_dir: Path = Path(args.save_dir) / args.experiment_name

    # NOTE: full_state_dict=False means we will be saving sharded checkpoints.
    ckpt_opts = StateDictOptions(full_state_dict=False, cpu_offload=True)

    # attempt resume
    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0,
    }
    resumed = False
    if (exp_dir / "state.json").exists():
        sharded_model_state, sharded_optimizer_state = get_state_dict(
            model, optimizer, options=ckpt_opts
        )
        load(
            dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
            checkpoint_id=exp_dir / "checkpoint",
        )
        set_state_dict(
            model,
            optimizer,
            model_state_dict=sharded_model_state,
            optim_state_dict=sharded_optimizer_state,
            options=ckpt_opts,
        )
        lr_scheduler.load_state_dict(
            torch.load(
                exp_dir / "lr_scheduler.pt", map_location=device, weights_only=True
            )
        )
        with open(exp_dir / "state.json") as fp:
            state = json.load(fp)
        resumed = True
    _LOGGER.info(f"Resumed={resumed} | {state}")

    dist.barrier()
    if rank == 0:
        # NOTE: assuming directory is shared across all nodes, that's why we do rank instead of local_rank
        _LOGGER.info(f"Creating experiment root directory")
        exp_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    (exp_dir / f"rank-{rank}").mkdir(parents=True, exist_ok=True)
    _LOGGER.info(f"Worker saving to {exp_dir / f'rank-{rank}'}")

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
        _LOGGER.info(f"Begin epoch {state['epoch']} at step {state['epoch_step']}")

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
                outputs = model(**batch)

            with timers["backward"]:
                optimizer.zero_grad(set_to_none=True)
                outputs.loss.backward()

            with timers["update"]:
                optimizer.step()
                lr_scheduler.step()

            state["global_step"] += 1
            state["epoch_step"] += 1
            state["running_loss"] += outputs.loss.item()
            progress_bar.update(1)

            if state["global_step"] % args.log_freq == 0:
                mem = torch.cuda.memory_stats(device)
                wandb.log(
                    {
                        "lr": lr_scheduler.get_last_lr()[0],
                        "running_loss": state["running_loss"] / args.log_freq,
                        "epoch": state["epoch"],
                        "epoch_progress": state["epoch_step"] / len(dataloader),
                        "num_batches_remaining": len(dataloader) - i_step,
                        f"curr_alloc_in_gb": 1e-9 * mem["allocated_bytes.all.current"],
                        f"peak_alloc_in_gb": 1e-9 * mem["allocated_bytes.all.peak"],
                        f"curr_resv_in_gb": 1e-9 * mem["reserved_bytes.all.current"],
                        f"peak_resv_in_gb": 1e-9 * mem["reserved_bytes.all.peak"],
                        "time/total": sum(t.avg_elapsed_ms() for t in timers.values()),
                        **{
                            f"time/{k}": timer.avg_elapsed_ms()
                            for k, timer in timers.items()
                        },
                    },
                    step=state["global_step"],
                )
                torch.cuda.reset_peak_memory_stats(device)
                state["running_loss"] = 0
                for t in timers.values():
                    t.reset()

            if state["global_step"] % args.ckpt_freq == 0:
                dist.barrier()
                # NOTE: we have to call this on ALL ranks
                sharded_model_state, sharded_optimizer_state = get_state_dict(
                    model, optimizer, options=ckpt_opts
                )
                save(
                    dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
                    checkpoint_id=exp_dir / "checkpoint",
                )
                if rank == 0:
                    torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
                    with open(exp_dir / "state.json", "w") as fp:
                        json.dump(state, fp)
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


@contextmanager
def rank0_first():
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()


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
    parser.add_argument(
        "--numel-to-wrap",
        default=100_000_000,
        type=int,
        help="Only applies FSDP to modules with numel > this value.",
    )
    parser.add_argument("--cpu-offload", default="off", choices=["on", "off"])
    parser.add_argument(
        "--bwd-prefetch",
        default="pre",
        choices=["BACKWARD_PRE", "BACKWARD_POST", "off"],
    )
    return parser


if __name__ == "__main__":
    main()
