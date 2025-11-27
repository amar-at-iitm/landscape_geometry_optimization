# Walkthrough

## Overview

This repository implements a **loss landscape geometry and optimization dynamics** framework for neural networks. It provides:
- Model definitions (MLP, ResNet) in `src/models/models.py`
- Training utilities in `src/training/trainer.py`
- Landscape analysis tools (visualization, Hessian metrics) in `src/landscape/`
- An end-to-end experiment script `experiments/train_and_analyze.py` that trains a model on synthetic data by default, computes Hessian eigenvalues, and generates 1-D/2-D visualizations.

## Setup

**It is better to create and activate a virtual python environment.

```bash
# Install required packages (torch, torchvision, etc.)
python -m pip install -r requirements.txt
```

## Running the Experiment

Default offline run (synthetic data + MLP):

```bash
python experiments/train_and_analyze.py --dataset synthetic --model mlp
```

Use CIFAR-10 instead (no download required when the cache in `data/` is present):

```bash
python experiments/train_and_analyze.py --dataset cifar10 --data_root data
```

Add `--download_data` if you need the script to fetch CIFAR-10 itself. Additional helpful flags:
- `--model resnet20` toggles to the residual network.
- `--save_dir <path>` changes where checkpoints and figures land (defaults to `experiments/outputs`).
- `--analysis_samples N` controls how many training examples power the Hessian/interpolation probes.

Each run performs the following stages:
1. **Data prep** - load the requested dataset with deterministic transforms for analysis.
2. **Model build** - instantiate `MLP` or `ResNet20`.
3. **Training** - run SGD and log train/test accuracy per epoch while saving the best checkpoint.
4. **Landscape probes** - compute the top Hessian eigenvalue, a 1-D interpolation, and a 2-D contour using a bounded subset of training samples.
5. **Artifact export** - write `model.pth`, `hessian.png`, `interp_1d.png`, and `contour_2d.png` into the output directory.

## Key Components

- **`src/models/models.py`** - Defines the `MLP` and `ResNet20` architectures used in experiments.
- **`src/training/trainer.py`** - Implements SGD training and evaluation helpers with checkpointing support.
- **`src/landscape/visualization.py`** - Provides direction normalization, interpolation utilities, and evaluation helpers for plotting.
- **`src/landscape/metrics.py`** - Contains the Hessian power-iteration routine based on double backpropagation.
- **`experiments/train_and_analyze.py`** - Orchestrates argument parsing, dataset construction, logging, and artifact creation.

## Extending the Framework

- **Add new models**: Introduce a class in `src/models/models.py` (mirroring `MLP`/`ResNet20`) and wire it into the CLI by extending the `--model` choices.
- **Custom datasets**: Modify `prepare_dataloaders` in `experiments/train_and_analyze.py` or add new dataset aliases exposed through `--dataset`.
- **Additional metrics**: Implement extra probes in `src/landscape/metrics.py` or visualization helpers and invoke them from the experiment driver.

## Verification

The integration test was run with synthetic data, and the pipeline completed without errors, producing:
- `model.pth` (trained checkpoint)
- Hessian eigenvalue logs
- 1-D and 2-D landscape visualizations

All components work together to provide a reproducible analysis of how model architecture and training dynamics affect the loss landscape.
