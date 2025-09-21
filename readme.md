# PIDReg: Explainable Multimodal Regression via Information Decomposition

## Overview

PIDReg is designed to fuse information from two modalities while automatically determining the optimal contribution of each modality through PID-based weight computation. The model learns to decompose the mutual information between modalities and the target variable into:
- **Unique Information**: Information exclusive to each modality
- **Redundant Information**: Overlapping information between modalities  
- **Synergistic Information**: Information that emerges only when both modalities are combined

## Key Features

- **Dynamic Fusion Weights**: Automatically computes fusion weights based on PID decomposition
- **Information Bottleneck**: Incorporates variational information bottleneck for each modality
- **Conditional Mutual Information Minimization**: Ensures modality-specific features remain informative
- **Adaptive Lambda Learning**: Learnable parameters control information bottleneck strength
- **PID Stability Detection**: Automatically fixes fusion weights when PID parameters stabilize


## Requirements

```bash
matplotlib==3.10.3
numpy==2.2.6
pandas==2.2.3
scikit_learn==1.6.1
scipy==1.15.3
torch==2.5.1+cu124
tqdm==4.66.2
```

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

This PIDReg Version expects CSV data with the following structure:
- Columns 0-80: Features for Modal 1
- Columns 81-166: Features for Modal 2  
- Column 167: Target variable

Example dataset: Superconductivity.csv

## Usage

### Basic Training

```bash
python main.py --data_path ./data --n_epochs 200 --batch_size 256
```

### Advanced Configuration

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

## Model Components

### PIDRegModel.py
Core model implementation featuring:
- Information bottleneck implementation
- PID-based fusion weight computation
- Gauss loss and CS divergence loss regularizations

### PIDRegTrainer.py
Training logic including:
- Separate optimizers for model and λ parameters
- Learning rate scheduling with ReduceLROnPlateau
- PID stability detection and automatic weight fixing

### CMICalculator.py
Conditional mutual information calculator using:
- CMI computation estimation

### csv_data_loader.py
Data loading utilities:
- Automatic train/validation/test splitting
- Standard scaling for features and labels