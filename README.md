# Distributed Training Tutorials

While there are many tutorials on how to do distributed training, there aren't any that compare and contrast the different approaches (e.g. torchrun, deepspeed, colossalai, etc).

Additionally, there is generally a lack of information on how any dependencies should be modified for use in distributed settings (e.g. wandb). This makes sense, because pytorch isn't really going to document how to best log things with wandb when using torchrun, and no one really thinks to check wandb documentation when thinking about distributed training.

Handling faults (software related issues, network issues, hardware, etc) during distributed training is a huge issue, and most tutorials don't go into best practices for how to identify what went wrong and how to fix it. This leads to huge wastes in resources unless you know all the best practices beforehand.

This tutorial is aimed at showing a realistic training setup and how it differs depending on the library used.

## Organization

Read through each of the READMEs in the subdirectories in order (starting with 01-single-gpu). Each readme will contain an overview of the differences between the last version.

There is a `train_llm.py` script in each directory which contains the training code for that setting.

**Each of the training scripts is aimed at training a causal language model (i.e. gpt).**

## Set up

### Virtual Environment

```bash
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
