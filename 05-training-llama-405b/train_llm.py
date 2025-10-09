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
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import (
    fully_shard,
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    FSDPModule,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.checkpoint import load, save

import tqdm
import datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)


LOGGER = logging.getLogger(__name__)


@record
def main():
    parser = _get_parser()
    args = parser.parse_args()

    rank = int(os.getenv("RANK", "0"))
    local_rank = rank % torch.cuda.device_count()
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(rank=rank, world_size=world_size, device_id=device)

    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    logging.basicConfig(
        format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )

    LOGGER.debug(os.environ)
    LOGGER.debug(args)
    LOGGER.debug(f"local_rank={local_rank} rank={rank} world size={world_size}")
    LOGGER.debug(f"Loading model from HF_HOME={os.getenv('HF_HOME')}")

    if args.cpu_offload:
        # NOTE: the optimizer will run on CPU when using CPU offloading,
        # so setting this will give us a little boost to that.
        torch.set_num_threads(os.cpu_count() // (torch.cuda.device_count()))

    # Get the full state dict on rank 0 so we can broadcast it later
    # with `set_model_state_dict()`
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

    # initialize model on meta device on **ALL** ranks
    model: torch.nn.Module
    with rank0_first(), torch.device("meta"):
        config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
        model = AutoModelForCausalLM.from_config(
            config,
            dtype=dtype,
            attn_implementation="flash_attention_2",
        )
    LOGGER.info(
        f"Training {sum(p.numel() for p in model.parameters())} model parameters"
    )

    # shard the models - nothing surprising here.
    fsdp_config = dict(
        reshard_after_forward=True,
        offload_policy=CPUOffloadPolicy() if args.cpu_offload else None,
        mp_policy=MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32),
    )
    for decoder in model.model.layers:
        fully_shard(decoder, **fsdp_config)
    fully_shard(model, **fsdp_config)
    LOGGER.debug("Sharded model")

    load_device = "cpu" if args.cpu_offload else device
    model.to_empty(device=load_device)
    dist.barrier()

    LOGGER.info(
        "Transferred model to device. Will now broadcast weights from rank 0..."
    )
    # Once we've sharded the meta model, we can allocate the tensors
    # and then load the model weights using the full_sd from rank 0
    # NOTE that the model here is the same on ALL ranks (even rank 0).
    # You might expect model on rank 0 to hold the weights,
    # but that is not what the API expects.
    set_model_state_dict(
        model=model,
        model_state_dict=full_sd,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            cpu_offload=args.cpu_offload,
        ),
    )

    # unfortunately `full_model.state_dict()`` won't contain
    # non-persistent buffers (i.e. `self.register_buffer(..., persistent=False)`),
    # so we have to manually broadcast them from `full_model` ourselves
    if rank == 0:
        for weight, buffer in zip(full_model.buffers(), model.buffers()):
            buffer.copy_(weight)
            dist.broadcast(buffer.to(device), src=0)
    else:
        for buffer in model.buffers():
            device_buffer = buffer.to(device)
            dist.broadcast(device_buffer, src=0)
            buffer.copy_(device_buffer.to(buffer.device))

    del full_sd
    # convienient way to force deallocation a model
    if rank == 0:
        full_model.to(torch.device("meta"))
    del full_model
    LOGGER.info(f"Initialized model uses {get_mem_stats(device)['curr_alloc_gb']}gb")

    if args.prefetch_layers:
        decoders = model.model.layers
        num_prefetch = 1
        for i, layer in enumerate(decoders):
            layer.set_modules_to_forward_prefetch(
                [
                    decoders[i + j]
                    for j in range(1, num_prefetch + 1)
                    if i + j < len(decoders)
                ]
            )
            layer.set_modules_to_backward_prefetch(
                [decoders[i - j] for j in range(1, num_prefetch + 1) if i - j >= 0]
            )

    # Applying gradient checkpointing - note that only the LlamaDecoderLayer supports this,
    # so we can just reuse our existing wrap_policy.
    if args.checkpoint_activations:
        from torch.nn import Embedding
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer, Embedding},
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            auto_wrap_policy=wrap_policy,
        )
        LOGGER.info("Applied gradient checkpoint")

    # NOTE: since this can download data, make sure to do the main process first on each node
    # since we manually specified HF_HOME to be a node local drive.
    with rank0_first():
        train_data = _load_and_preprocess_data(args, config)
    LOGGER.debug(f"{len(train_data)} training samples")

    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=1,
        prefetch_factor=2,
        # NOTE: this sampler will split dataset evenly across workers
        sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
    )
    LOGGER.debug(f"{len(dataloader)} batches per epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=args.lr * 1e-2
    )

    model = torch.compile(model)
    model.loss_function = torch.compile(model.loss_function)
    optimizer.step = torch.compile(optimizer.step)
    LOGGER.info("Compiled model")

    is_experiment = False
    exp_dir: Path = Path(args.save_dir)
    if args.experiment_name is not None:
        is_experiment = True
        exp_dir = exp_dir / args.experiment_name

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
    if is_experiment and (exp_dir / "state.json").exists():
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
    if is_experiment:
        LOGGER.info(f"Resumed={resumed} | {state}")
    dist.barrier()

    if is_experiment and (
        (exp_dir.is_mount() and rank == 0)
        or (not exp_dir.is_mount() and local_rank == 0)
    ):
        LOGGER.info(f"Creating experiment root directory")
        exp_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    timers = {k: LocalTimer(device) for k in ["data", "forward", "backward", "update"]}

    for state["epoch"] in range(state["epoch"], args.num_epochs):
        LOGGER.info(f"Begin epoch {state['epoch']} at step {state['epoch_step']}")

        progress_bar = tqdm.tqdm(range(len(dataloader)))
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])

        dataloader.sampler.set_epoch(state["epoch"])
        batches = iter(dataloader)

        for i_step in range(len(dataloader)):
            with timers["data"], torch.no_grad():
                batch = next(batches)
                batch = {k: v.to(device=device) for k, v in batch.items()}

            if i_step < state["epoch_step"]:
                # NOTE: for resuming
                continue

            torch.cuda.empty_cache()
            with timers["forward"]:
                outputs = model(**batch)

            torch.cuda.empty_cache()
            with timers["backward"]:
                outputs.loss.backward()

            torch.cuda.empty_cache()
            with timers["update"]:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            state["global_step"] += 1
            state["epoch_step"] += 1
            state["running_loss"] += outputs.loss.item()
            progress_bar.update(1)

            if state["global_step"] % args.log_freq == 0:
                tok_per_step = world_size * args.batch_size * args.seq_length
                ms_per_step = sum(t.avg_elapsed_ms() for t in timers.values())
                info = {
                    "global_step": state["global_step"],
                    "lr": lr_scheduler.get_last_lr()[0],
                    "running_loss": state["running_loss"] / args.log_freq,
                    "epoch": state["epoch"],
                    "epoch_progress": state["epoch_step"] / len(dataloader),
                    "num_batches_remaining": len(dataloader) - i_step,
                    **get_mem_stats(device),
                    "tokens_per_s": 1000 * tok_per_step / ms_per_step,
                    "time/total": ms_per_step,
                    "time/total": sum(t.avg_elapsed_ms() for t in timers.values()),
                    **{
                        f"time/{k}": timer.avg_elapsed_ms()
                        for k, timer in timers.items()
                    },
                }

                LOGGER.info(info)

                torch.cuda.reset_peak_memory_stats(device)
                state["running_loss"] = 0
                for t in timers.values():
                    t.reset()

            if is_experiment and state["global_step"] % args.ckpt_freq == 0:
                LOGGER.info("Saving checkpoint.")
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


def _load_and_preprocess_data(args, config):
    """
    Function created using code found in
    https://github.com/huggingface/transformers/blob/v4.45.1/examples/pytorch/language-modeling/run_clm_no_trainer.py
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    data = datasets.load_dataset(args.dataset_name, args.dataset_subset)

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

    seq_length = args.seq_length or tokenizer.model_max_length
    if seq_length > config.max_position_embeddings:
        seq_length = min(1024, config.max_position_embeddings)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        if total_length > seq_length:
            total_length = (total_length // seq_length) * seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {seq_length}",
    )

    return lm_datasets["train"]


def get_mem_stats(device=None):
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "total_gb": 1e-9 * props.total_memory,
        "curr_alloc_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "curr_resv_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_resv_gb": 1e-9 * mem["reserved_bytes.all.peak"],
    }


@contextmanager
def rank0_first():
    rank = dist.get_rank() % torch.cuda.device_count()
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
    parser.add_argument("-e", "--experiment-name", default=None)
    parser.add_argument("-d", "--dataset-name", default=None, required=True)
    parser.add_argument("--dataset-subset", default=None)
    parser.add_argument("-m", "--model-name", default=None, required=True)
    parser.add_argument("--save-dir", default="../outputs")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("--log-freq", default=100, type=int)
    parser.add_argument("--ckpt-freq", default=500, type=int)
    parser.add_argument("-s", "--seq-length", default=1024, type=int)
    parser.add_argument("--cpu-offload", default=False, action="store_true")
    parser.add_argument("--checkpoint-activations", default=False, action="store_true")
    parser.add_argument("--prefetch-layers", default=False, action="store_true")
    return parser


if __name__ == "__main__":
    main()
