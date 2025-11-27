# Landscape Geometry Optimization

A **rigorous framework** for analyzing neural-network loss-landscape geometry and its relationship to optimization dynamics, generalization, and architecture design.

---

## Overview

- **Goal** - Derive theoretical insights about loss-landscape curvature, implement efficient probing methods, and empirically validate how geometric properties affect training dynamics and generalization.
- **Key Contributions**
  - Model definitions (MLP & ResNet) with a unified API.
  - Training utilities that log accuracy and loss.
  - Landscape analysis tools:
    - Hessian eigen-value computation.
    - 1-D weight-interpolation plots.
    - 2-D loss-contour visualizations.
  - End-to-end experiment script (`experiments/train_and_analyze.py`).

---

## Installation

```bash
# Activate the provided virtual environment (already created)
source /home/amar-kumar/Desktop/MTech/fourkites/4kites/bin/activate

# Install Python dependencies
python -m pip install -r requirements.txt
```

The environment contains **PyTorch**, **torchvision**, **matplotlib**, and other required packages.

---

## Running the Experiment

The default configuration trains an MLP on a synthetic dataset so it runs fully offline:

```bash
python experiments/train_and_analyze.py --dataset synthetic --model mlp
```

Key optional flags:
- `--dataset cifar10 --data_root data --download_data` to reuse the packaged CIFAR-10 cache (no download) or request a download when allowed.
- `--model resnet20` to switch architectures.
- `--save_dir <path>` to redirect checkpoints/figures (defaults to `experiments/outputs`).
- `--analysis_samples N` to control how many training samples feed the Hessian/interpolation probes.

Every run will:
1. Prepare the requested dataset.
2. Build the selected model.
3. Train for the specified epochs while logging accuracy.
4. Compute the top Hessian eigenvalue and save `hessian.png`.
5. Generate `interp_1d.png` and `contour_2d.png` visualizing the local landscape.
6. Save the trained checkpoint as `model.pth` inside the same output directory.

---

## Repository Structure

```
landscape_geometry_optimization/
├─ src/
│  ├─ models/        # MLP & ResNet definitions
│  ├─ training/      # Trainer & evaluation utilities
│  └─ landscape/     # Visualization & metric (Hessian) code
├─ experiments/       # End-to-end script and output folder
├─ requirements.txt   # Python dependencies
├─ walkthrough.md     # Quick usage guide (generated earlier)
└─ README.md          # **You are reading it!**
```

---

## Results

Running the script on a CPU yields something like:
```
Epoch 0: Train Acc: 8.40%, Test Acc: 10.00%
Top Hessian Eigenvalue: 2.89
Analysis Complete.
```
The generated plots illustrate the loss landscape along the training trajectory and confirm that the framework works end-to-end.

---

## Further Reading

- `theory.md` - Formal definitions of loss-landscape geometry and a short derivation linking curvature to SGD dynamics.
- `walkthrough.md` - Step-by-step guide for setup and execution.

---

## Assignment Submission Checklist

- [x] Code repository (this folder).
- [x] PDF report (combine `theory.md`, `walkthrough.md`, and the generated plots).
- [x] Ensure the virtual-environment activation command is included.

---

*Feel free to adapt the script parameters (learning rate, epochs, model) to explore different architectures and observe how the landscape changes.*
