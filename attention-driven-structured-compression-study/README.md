# Attention-Driven Structured Compression Study

A study of **Attention-Driven Structured Compression (ADSC)** — using learned attention weights to selectively prune channels and layers in neural networks, preserving accuracy while reducing model size.

Explored across three datasets: MNIST, CIFAR-10, and California Housing (regression).

## How It Works

Instead of pruning by magnitude or randomly, ADSC learns an **attention权重** for each channel/layer during training. Channels with low attention are pruned; channels with high attention are preserved. This allows the network to decide which parts are important.

### Compression Loss

The compression objective combines:
- **Task loss** (cross-entropy or MSE) — maintain prediction quality
- **Attention sparsity loss** — L1 regularization on attention weights to encourage pruning
- **Variance diversity loss** — encourage diverse attention across layers

## Experiments

Three benchmarks, each with a teacher-student setup:

### MNIST (Classification)
- Teacher: standard CNN → Student: compressed via ADSC
- Compares accuracy vs. uniform pruning and no-attention baselines

### CIFAR-10 (Classification)
- Same teacher-student compression approach

### California Housing (Regression)
- 1D CNN adapted for regression
- ADSC applied with 1D-specific adjustments

## Key Comparisons

| Variant | Description |
|---|---|
| **Full ADSC** | Attention-driven compression with sparsity + variance loss |
| Uniform pruning | Global uniform compression ratio across all layers |
| No attention | Plain mean activation pruning without learnable attention |

## Files

- `adsc_study.ipynb` — Full notebook with experiments
- `teacher_*.pt` / `student_*.pt` — Trained model weights for each dataset

## Libraries

`PyTorch`, `matplotlib`, `numpy`, `scikit-learn`
