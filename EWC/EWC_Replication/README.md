# EWC Replication

This repository contains a PyTorch implementation of Elastic Weight Consolidation (EWC) as described in "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al., 2017).

## Setup
```bash
pip install -r requirements.txt
```

## Results 

Results
Our implementation achieves comparable results to the original paper:

Task 1: 96.33% accuracy
Task 2: 96.47% accuracy (maintained)
Task 3: 96.83% accuracy (maintained)
Implementation Details
Model: 2-layer MLP (400 hidden units)
Dataset: Permuted MNIST
EWC λ: 400
Optimizer: SGD (lr=0.1)

## Implementation Details 

- Model: 2-layer MLP (400 hidden units)
- Dataset: Permuted MNIST
- EWC λ: 400
- Optimizer: SGD (lr=0.1)