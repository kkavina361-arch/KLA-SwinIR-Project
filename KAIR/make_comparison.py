import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ============================================================
# IMAGE
# ============================================================

filename = "000002.npy"

# ============================================================
# PATHS
# ============================================================

noisy_path = f"../datasets/validation/NoisyLR/{filename}"
gt_path = f"../datasets/validation/GT/{filename}"
output_path = f"../results/validation_65000_E/{filename}"

save_path = "../results/comparison_000002.png"

# ============================================================
# LOAD
# ============================================================

noisy = np.load(noisy_path).astype(np.float32)
gt = np.load(gt_path).astype(np.float32)
output = np.load(output_path).astype(np.float32)

noisy = np.squeeze(noisy)
gt = np.squeeze(gt)
output = np.squeeze(output)

# ============================================================
# UPSCALE NOISY IMAGE FOR VISUAL COMPARISON ONLY
# ============================================================

noisy_display = resize(
    noisy,
    gt.shape,
    order=1,
    preserve_range=True,
    anti_aliasing=False
)

# ============================================================
# METRICS
# ============================================================

psnr = peak_signal_noise_ratio(
    gt,
    output,
    data_range=1.0
)

ssim = structural_similarity(
    gt,
    output,
    data_range=1.0
)

print("==========================================")
print("IMAGE COMPARISON")
print("==========================================")
print("Image :", filename)
print(f"PSNR  : {psnr:.4f} dB")
print(f"SSIM  : {ssim:.4f}")
print("==========================================")

# ============================================================
# DISPLAY
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13, 4.5)
)

axes[0].imshow(
    np.clip(noisy_display, 0, 1),
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[0].set_title(
    "Noisy Input\n128 × 128"
)

axes[1].imshow(
    np.clip(output, 0, 1),
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[1].set_title(
    f"SwinIR Output\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.3f}"
)

axes[2].imshow(
    np.clip(gt, 0, 1),
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "Ground Truth\n256 × 256"
)

for ax in axes:
    ax.axis("off")

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print()
print("Comparison saved to:")
print(save_path)

plt.show()