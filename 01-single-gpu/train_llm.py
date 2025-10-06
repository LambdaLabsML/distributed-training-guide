import argparse
from itertools import chain
import json
import multiprocessing
import os
import time
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
import tqdm
import datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

LOGGER = logging.getLogger(__name__)


def main():
    parser = _get_parser()
    args = parser.parse_args()

    # Will be modifying this in future version to include rank information
    logging.basicConfig(
        format=f"[%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )

    # Helpful to log this information when running on multiple nodes to make sure all nodes have the same environment.
    LOGGER.debug(os.environ)
    LOGGER.debug(args)

    # This guide assumes CUDA device is available, and does all training in bf16
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Seed pytorch's RNG. See https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.seed)

    # Note: Initializing an **untrained** model
    config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
    with device:
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    LOGGER.info(f"Training {sum(p.numel() for p in model.parameters())} model parameters")

    train_data = _load_and_preprocess_data(args, config)
    LOGGER.debug(f"{len(train_data)} training samples")

    # Standard pytorch dataset iterator
    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=default_data_collator,
    )
    LOGGER.info(f"{len(dataloader)} batches per epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)

    # NOTE: T_max and eta_min were arbitrarily chosen
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=args.lr * 1e-2
    )

    exp_dir: Path = Path(args.save_dir) / args.experiment_name
    LOGGER.info(f"Experiment saving to {exp_dir}")

    # attempt resume
    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0,
    }
    resumed = False
    if (exp_dir / "state.json").exists():
        # NOTE: weights_only is to protect against arbitrary code execution with pickle decoding.
        def _load_to_device(p):
            return torch.load(p, map_location=device, weights_only=True)

        model.load_state_dict(_load_to_device(exp_dir / "model.pt"))
        optimizer.load_state_dict(_load_to_device(exp_dir / "optimizer.pt"))
        lr_scheduler.load_state_dict(_load_to_device(exp_dir / "lr_scheduler.pt"))
        with open(exp_dir / "state.json") as fp:
            state = json.load(fp)
        resumed = True
    LOGGER.info(f"Resumed={resumed} | {state}")

    LOGGER.info(f"Creating experiment root directory")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # will be using to understand breakdown of speed
    timers = {k: LocalTimer(device) for k in ["data", "forward", "backward", "update"]}

    for state["epoch"] in range(state["epoch"], args.num_epochs):
        LOGGER.info(f"Begin epoch {state['epoch']} at step {state['epoch_step']}")

        progress_bar = tqdm.tqdm(range(len(dataloader)))
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])

        # NOTE: This is not standard. Normally you can just iterate directly over dataloader.
        #       We are doing this so we can explicitly measure the time it takes to generate a batch.
        batches = iter(dataloader)

        for i_step in range(len(dataloader)):
            # Here we measure the time it takes to generate a batch and move it to the GPU
            with timers["data"], torch.no_grad():
                batch = next(batches)
                batch = {k: v.to(device=device) for k, v in batch.items()}

            # For resuming, this has to come after getting the next batch, so we move through the dataset properly.
            if i_step < state["epoch_step"]:
                # NOTE: for resuming
                continue

            with timers["forward"]:
                outputs = model(**batch)

            with timers["backward"]:
                # NOTE: set_to_none=True will de-allocate the gradients, saving us some memory.
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
                tok_per_step = args.batch_size * args.seq_length
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

            if state["global_step"] % args.ckpt_freq == 0:
                LOGGER.info("Saving checkpoint.")
                torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
                torch.save(model.state_dict(), exp_dir / "model.pt")
                torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
                with open(exp_dir / "state.json", "w") as fp:
                    json.dump(state, fp)

        state["epoch_step"] = 0


def _load_and_preprocess_data(args, config):
    """
    Function created using code found in
    https://github.com/huggingface/transformers/blob/v4.45.1/examples/pytorch/language-modeling/run_clm_no_trainer.py
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    data = datasets.load_dataset(args.dataset_name, trust_remote_code=True)

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
    parser.add_argument("-e", "--experiment-name", default=None, required=True)
    parser.add_argument("-d", "--dataset-name", default=None, required=True)
    parser.add_argument("-m", "--model-name", default=None, required=True)
    parser.add_argument("--save-dir", default="../outputs")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("--log-freq", default=100, type=int)
    parser.add_argument("--ckpt-freq", default=500, type=int)
    parser.add_argument("-s", "--seq-length", default=1024, type=int)
    return parser


if __name__ == "__main__":
    main()
