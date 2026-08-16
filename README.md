\# AI-Based Restoration of Degraded Semiconductor Inspection Images



\## KLA Hackathon – PS01



This project presents an AI-based image restoration pipeline designed for degraded grayscale semiconductor inspection images. The objective is to recover clean, high-resolution images from inputs affected by noise and spatial resolution reduction while preserving fine structural details that may be important during semiconductor inspection.



The solution is based on a lightweight customized \*\*SwinIR (Swin Transformer for Image Restoration)\*\* model and performs \*\*2× super-resolution and restoration in a single inference pipeline\*\*.



\---



\## Problem Statement



Semiconductor inspection relies on high-quality microscopic images for identifying and analysing small structures and potential defects.



In practical imaging conditions, inspection images may suffer from:



\- Speckle noise

\- Gaussian noise

\- Loss of fine structural information

\- Spatial resolution reduction

\- Combined noise and downsampling



The challenge is to reconstruct the corresponding clean, high-resolution image while avoiding excessive smoothing or artificial image structures.



The model therefore learns the mapping:



Degraded Low-Resolution Image → Restored High-Resolution Image



For the submitted configuration:



\- Input: `128 × 128` grayscale image

\- Output: `256 × 256` restored grayscale image

\- Upscaling factor: `2×`



\---



\## Proposed Approach



We use a customized \*\*SwinIR\*\* architecture based on Swin Transformer blocks.



Instead of performing denoising and super-resolution as completely separate stages, the network learns a direct transformation from the degraded low-resolution input to the clean high-resolution target.



The submitted model configuration uses:



\- Input channels: `1`

\- Upscaling factor: `2`

\- Window size: `8`

\- Embedding dimension: `60`

\- Transformer depths: `\[2, 2, 2, 2]`

\- Attention heads: `\[3, 3, 3, 3]`

\- MLP ratio: `2`

\- Upsampler: `PixelShuffle`

\- Residual connection: `1conv`



This compact configuration was selected to balance restoration quality and inference efficiency.



\---



\## Training Strategy



The model was trained using paired degraded and ground-truth grayscale images.



The final training configuration used:



\- Maximum iterations: `1000`

\- Learning rate: `1e-5`

\- Random seed: `42`

\- Optimizer: Adam

\- Training input: degraded low-resolution images

\- Training target: clean high-resolution ground-truth images



\### Loss Function



The final model combines pixel reconstruction and structural similarity objectives.



The training objective uses:



\*\*L1 reconstruction loss + SSIM-based structural loss\*\*



with:



\- SSIM weight: `0.1`



L1 loss encourages accurate pixel-level reconstruction, while the SSIM component encourages preservation of structural information and local image characteristics.



Data augmentation is also used during training to improve robustness and reduce dependence on individual training-image orientations.



\---



\## Repository Structure



```text

KLA-SwinIR-Project/

│

├── evaluate.py

├── requirements.txt

├── README.md

│

├── checkpoints/

│   └── l1\_ssim\_aug\_1000iter\_G.pth

│

├── KAIR/

│   ├── models/

│   │   └── network\_swinir.py

│   ├── train\_l1ssim\_aug\_cpu.py

│   └── ...

│

└── submission/

&#x20;   └── test\_outputs/

&#x20;       ├── 000000.npy

&#x20;       ├── 000000.png

&#x20;       ├── ...

&#x20;       ├── 000399.npy

&#x20;       └── 000399.png

