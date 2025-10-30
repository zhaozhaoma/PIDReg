# PIDReg: Explainable Multimodal Regression via Information Decomposition

## Overview
<p align="center">
  <img src="https://github.com/zhaozhaoma/Images/blob/main/ICLR2026/Flow.png" width="1000" alt="Overview"/>
</p>

| |
|:--:|
| *Framework of **P**artial **I**nformation **D**ecomposition for Multimodal **Reg**ression (**PIDReg**), illustrated with video and audio modalities, where $`P(X_{1})`$, $`P(X_{2})`$, and $`P(Y)`$ denote empirical data distributions that may deviate from Gaussianity (e.g., skewed or heavy-tailed).* |

PIDReg adaptively fuses two input modalities by learning how much each modality should contribute to prediction, rather than relying on fixed or hand-tuned weights. During training, PIDReg operates in the joint latent space of the modalities and decomposes the information into:

- 🔺 **Unique Information**  
  Modality-specific information that is captured by one modality but not the other.

- 🔶 **Redundant Information**  
  Overlapping information between modalities.

- 🔴 **Synergistic Information**  
  Information that emerges when both modalities are considered jointly, interactions that neither modality can provide alone.

By explicitly estimating these components, PIDReg can (i) quantify what each modality is truly contributing, and (ii) weight modalities accordingly during regression.

##  Key Features

| 🌈 | Feature | Description |
|:--:|:--|:--|
| ① | **Dynamic Fusion Weights** | Learns optimal fusion weights via PID-based decomposition, eliminating hand-crafted balancing. |
| ② | **Information Bottleneck** | Enforces compact yet informative latent representations for each modality. |
| ③ | **Conditional MI Minimization** | Minimizes redundancy to enhance modality-specific informativeness. |
| ④ | **Adaptive λ-Learning** | Automatically adjusts information bottleneck strength through learnable λ parameters. |
| ⑤ | **PID Stability Detection** | Detects stabilization of PID terms and freezes weights to prevent overfitting. |


## Installation

1. Repository preparation:
```bash
cd PIDReg
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have CUDA-compatible PyTorch installed for GPU acceleration

## Data Format

To make PIDReg directly runnable, we include a curated sample dataset (`./data/sample.csv`). This file contains 5,000 rows subsampled from the original *Superconductivity* dataset used in our experiments. Each row in `sample.csv` follows a unified and intuitive column layout:

| Index Range | Description | Modality |
|:--:|:--|:--:|
| 0–80 | Feature dimensions for the first input source | **Modality 1** |
| 81–166 | Feature dimensions for the second input source | **Modality 2** |
| 167 | Continuous target variable used for regression | **Target Variable** |

## Model Components

| File | Purpose | Key Functionalities |
|:--|:--|:--|
| **`PIDRegModel.py`** | Core model definition | • Implements **modality-specific information bottlenecks**  <br> • Computes **PID-based fusion weights** for adaptive modality contribution  <br> • Incorporates **Gaussian reconstruction** and **Cauchy–Schwarz divergence** regularizations |
| **`PIDRegTrainer.py`** | Training orchestration | • Uses **dual optimizers** (one for model parameters, one for λ)  <br> • Integrates a scheduler for adaptive learning rate control  <br> • Performs **PID stability detection** and **automatic fusion-weight freezing** once convergence reached |
| **`CMICalculator.py`** | Conditional mutual information module | • Estimates **conditional mutual information (CMI)** between modalities and targets  <br> • Guides training toward **reducing information leakage** and enhancing modality specificity |
| **`csv_data_loader.py`** | Data processing and preparation | • Automatically splits data into **train / validation / test** sets  <br> • Applies **standard scaling** to both features and target  <br> • Supports flexible dataset paths and batch construction |

## Usage

1. Basic Training

```bash
python main.py --data_path ./data --n_epochs 200 --batch_size 256
```

2. Advanced Configuration

```bash
python main.py \
    --data_path ./data \
    --result_dir ./results \
    --batch_size 256 \
    --n_epochs 200 \
    --window_size 5 \
    --early_stopping 30 \
    --lambda_lr 0.1 \
    --hidden_dim 256 \
    --latent_dim 64
```

## Citation

If you utilize PIDReg in your work, we would appreciate your citation of the following paper📃:

```
@article{,
  title={Explainable Multimodal Regression via Information Decomposition},
  author={Zhaozhao Ma and Shujian YU},
  journal={},
  year={}
}
```

## Contact

For any questions or feedback, please feel free to reach out to us via email: `zhaozhaoma@gatech.edu`

---

<div align="center">

Built with ❤️, and ☀️🌙

</div>

---

