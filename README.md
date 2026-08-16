# KLA SwinIR

### AI-Based Restoration of Degraded Semiconductor Inspection Images

A SwinIR-based image restoration pipeline developed for the **KLA Hackathon PS01** challenge.

The project focuses on restoring degraded grayscale semiconductor inspection images while performing **2× super-resolution** in a single inference pipeline.

**Input:** 128 × 128 grayscale
**Output:** 256 × 256 grayscale
**Upscaling:** 2×
**Model:** Customized SwinIR

---

## Overview

High-quality inspection images are important for preserving small structures and visual details during semiconductor analysis.

The project focuses on restoring degraded grayscale semiconductor inspection images affected by noise and reduced spatial resolution.

In practical imaging conditions, images can be affected by:

- Speckle noise
- Gaussian noise
- Reduced spatial resolution
- Loss of fine structural information
- Combined noise and downsampling degradation

This project addresses the restoration problem by learning a direct mapping from a degraded low-resolution image to its clean high-resolution counterpart.

### Target Configuration

| Property | Value |
|---|---|
| Input | 128 × 128 grayscale |
| Output | 256 × 256 grayscale |
| Upscaling | 2× |
| Model | Customized SwinIR |
| Framework | PyTorch |

---

## Approach

The project uses a customized **SwinIR (Swin Transformer for Image Restoration)** architecture.

SwinIR uses hierarchical Transformer blocks with shifted-window attention to model local image structures efficiently. For this project, the architecture was configured as a compact restoration model for grayscale 2× super-resolution.

### Processing Flow

```text
Degraded LR Image
       │
       ▼
Input Processing
       │
       ▼
Customized SwinIR
       │
       ├── Swin Transformer Blocks
       ├── Shifted-Window Attention
       ├── Residual Learning
       └── PixelShuffle Upsampling
       │
       ▼
Restoration + 2× Super-Resolution
       │
       ▼
Restored HR Image
       │
       ├── .npy → Float32 benchmark output
       └── .png → Visual output
```

The model performs restoration and 2× super-resolution within a single inference pipeline.

---

## Model Configuration

| Parameter | Value |
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

The configuration was selected to balance restoration capability and inference efficiency.

---

## Training

The model was trained using paired degraded low-resolution images and clean high-resolution ground-truth images.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Maximum iterations | 1000 |
| Learning rate | 1e-5 |
| Random seed | 42 |
| Input | Degraded low-resolution images |
| Target | Clean high-resolution images |

Data augmentation was used during training to improve robustness.

### Loss Function

The final training setup combines L1 reconstruction loss with an SSIM-based structural loss.

```
Total Loss = L1 Loss + 0.1 × SSIM Loss
```

L1 loss helps maintain pixel-level accuracy, while the SSIM component encourages preservation of structural information.

### Final Model

The submitted checkpoint is:

```
checkpoints/l1_ssim_aug_1000iter_G.pth
```

**Checkpoint size:** 18.54 MB

The checkpoint is loaded directly by the evaluation pipeline.

---

## Evaluation

The repository includes a standalone `evaluate.py` script.

The evaluation pipeline:

1. Loads the trained SwinIR checkpoint.
2. Reads the degraded input images.
3. Runs model inference.
4. Generates restored floating-point outputs.
5. Saves `.npy` and `.png` results.
6. Reports inference timing.

The script uses CUDA when a CUDA-enabled PyTorch environment is available and otherwise falls back to CPU.

### Local Verification

The evaluation pipeline was tested in a separate environment before submission.

| Metric | Result |
|---|---|
| Images processed | 400 |
| NPY outputs | 400 |
| PNG outputs | 400 |
| Failed images | 0 |

**Total inference time:** 206.2167 s
**Average per image:** 515.542 ms

> The reported timing is from the local CPU verification environment and is not an H100 benchmark.

---

## Running the Evaluation

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the evaluation:

```bash
python evaluate.py --model_path checkpoints/l1_ssim_aug_1000iter_G.pth --input_dir path/to/test/NoisyLR --output_dir restored_outputs
```

### Output Format

| Format | Purpose |
|---|---|
| `.npy` | Float32 output for quantitative benchmarking |
| `.png` | Visual output for inspection |

**Input:** 128 × 128 grayscale
**Output:** 256 × 256 grayscale
**Scale:** 2×

---

## Repository Structure

```
KLA-SwinIR-Project/
│
├── README.md
├── requirements.txt
├── evaluate.py
│
├── checkpoints/
│   └── l1_ssim_aug_1000iter_G.pth
│
├── KAIR/
│   ├── models/
│   │   ├── network_swinir.py
│   │   └── loss_ssim.py
│   └── train_l1ssim_aug_cpu.py
│
├── configs/
├── scripts/
│
└── submission/
    └── test_outputs/
        ├── 000000.npy
        ├── 000000.png
        ├── ...
        ├── 000399.npy
        └── 000399.png
```

---

## Dependencies

The main evaluation dependencies are:

- numpy
- opencv-python
- timm
- torch
- torchvision

They are listed in `requirements.txt`.

For GPU execution, PyTorch should be installed with a CUDA configuration compatible with the target GPU and NVIDIA driver environment.

---

## Reproducibility

The repository contains the main components required to reproduce the submitted inference pipeline:

- Final trained checkpoint
- Standalone evaluation script
- SwinIR model implementation
- SSIM loss implementation
- Training script
- Dependency specification
- Test outputs for verification

The original training and validation datasets are not included in the repository.

---

## Project Highlights

- Customized SwinIR restoration model
- 2× super-resolution
- Grayscale image processing
- L1 + SSIM training objective
- Data augmentation
- Standalone evaluation pipeline
- CUDA-aware inference
- Fresh-environment verification
- Float32 `.npy` benchmark outputs
- `.png` visual outputs
- 18.54 MB final checkpoint

---

## References

This project builds on:

- **SwinIR** — Swin Transformer for Image Restoration
- **KAIR** — Image Restoration Toolbox
- **PyTorch**
- **KLA Hackathon – PS01**

**Task:** AI-Based Restoration of Degraded Images for Semiconductor Inspection
**Model:** Customized SwinIR
**Input:** 128 × 128 grayscale
**Output:** 256 × 256 grayscale
**Upscaling:** 2×
**Final Checkpoint:** `l1_ssim_aug_1000iter_G.pth`
