# Distributed Training Guide

This guide aims at a comprehensive guide on best practices for distributed training, diagnosing errors, and fully utilize all resources available.

## Questions this guide answers:

- How do I update a single gpu training/fine tuning script to run on multiple GPUs or multiple nodes?
- How do I diagnose hanging/errors that happen during training?
- My model/optimizer is too big for a single gpu - how do I train/fine tune it on my cluster?
- How do I schedule/launch training on a cluster?
- How do I scale my hyperparameters when increasing the number of workers?

---

Best practices for logging stdout/stderr and wandb are also included, as logging is vitally important in diagnosing/debugging training runs on a cluster.

## How to read

This guide is organized into sequential chapters, each with a `README.md` and a `train_llm.py` script in them. The readme will discuss the changes introduced in that chapter, and go into more details.

**Each of the training scripts is aimed at training a causal language model (i.e. gpt).**

## Set up

### Clone this repo

```bash
git clone https://github.com/LambdaLabsML/distributed-training-guide.git
```

### Virtual Environment

```bash
cd distributed-training-guide
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
pip install -U setuptools wheel
pip install -r requirements.txt
```

### wandb

This tutorial uses `wandb` as an experiment tracker.

```bash
wandb login
```

<p align="center">
🦄 Other exciting ML projects at Lambda: <a href="https://news.lambdalabs.com/news/today">ML Times</a>, <a href="https://lambdalabsml.github.io/Open-Sora/introduction/">Text2Video</a>, <a href="https://lambdalabs.com/gpu-benchmarks">GPU Benchmark</a>.
</p>
