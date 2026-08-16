# KLA SwinIR – Semiconductor Image Restoration

## KLA Hackathon – PS01

An AI-based image restoration pipeline for degraded grayscale semiconductor inspection images, built around a customized SwinIR model for 2× super-resolution and image restoration.

---

## Overview

Semiconductor inspection depends on high-quality images to preserve small structures and important visual details.

In practical imaging conditions, images can be affected by:

- Speckle noise
- Gaussian noise
- Reduced spatial resolution
- Loss of fine structural information
- Combined noise and downsampling degradation

This project addresses the restoration problem by learning a direct mapping from a degraded low-resolution image to its clean high-resolution counterpart.

**Input:** 128 × 128 grayscale image  
**Output:** 256 × 256 restored grayscale image  
**Upscaling:** 2×

---

## Approach

The project uses a customized **SwinIR (Swin Transformer for Image Restoration)** architecture.

SwinIR uses hierarchical Transformer blocks with shifted-window attention to model both local and broader image structures. For this project, the architecture was configured as a compact restoration model suitable for the target task.

### Model Configuration

| Parameter | Configuration |
|---|---|
| Input channels | 1 |
| Upscaling factor | 2× |
| Window size | 8 |
| Embedding dimension | 60 |
| Transformer depths | [2, 2, 2, 2] |
| Attention heads | [3, 3, 3, 3] |
| MLP ratio | 2 |
| Upsampler | PixelShuffle |
| Residual connection | 1conv |

The model performs restoration and 2× super-resolution within a single inference pipeline.

---

## Training

The model was trained using paired degraded low-resolution images and clean high-resolution ground-truth images.

### Training Configuration

- Optimizer: Adam
- Maximum iterations: 1000
- Learning rate: 1e-5
- Random seed: 42
- Input: degraded low-resolution images
- Target: clean high-resolution images

### Loss Function

The final training setup combines:

**L1 reconstruction loss + SSIM-based structural loss**

The SSIM contribution uses a weight of **0.1**.

L1 loss helps maintain pixel-level accuracy, while the SSIM component encourages preservation of structural information.

Data augmentation was also used during training to improve robustness.

---

## Evaluation

The repository includes a standalone evaluation script:

```text
evaluate.py