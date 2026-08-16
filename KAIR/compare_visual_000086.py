import os
import numpy as np
import matplotlib.pyplot as plt

filename = "000086.npy"

noisy_path = rf"D:\KLA_SwinIR_Project\datasets\validation\NoisyLR\{filename}"
gt_path = rf"D:\KLA_SwinIR_Project\datasets\validation\GT\{filename}"

baseline_path = rf"D:\results\validation_100000_E_cpu\{filename}"
l1ssim_path = rf"D:\KLA_SwinIR_Project\results\validation_l1ssim_100_cpu\{filename}"

output_path = rf"D:\KLA_SwinIR_Project\results\comparison_L1_vs_L1SSIM_{filename[:-4]}.png"

noisy = np.squeeze(np.load(noisy_path).astype(np.float32))
gt = np.squeeze(np.load(gt_path).astype(np.float32))
baseline = np.squeeze(np.load(baseline_path).astype(np.float32))
l1ssim = np.squeeze(np.load(l1ssim_path).astype(np.float32))

noisy = np.clip(noisy, 0.0, 1.0)
baseline = np.clip(baseline, 0.0, 1.0)
l1ssim = np.clip(l1ssim, 0.0, 1.0)
gt = np.clip(gt, 0.0, 1.0)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(noisy, cmap="gray")
axes[0].set_title("Degraded Input\n128×128")
axes[0].axis("off")

axes[1].imshow(baseline, cmap="gray")
axes[1].set_title("100K L1\n256×256")
axes[1].axis("off")

axes[2].imshow(l1ssim, cmap="gray")
axes[2].set_title("L1 + SSIM\n100 iterations")
axes[2].axis("off")

axes[3].imshow(gt, cmap="gray")
axes[3].set_title("Ground Truth\n256×256")
axes[3].axis("off")

plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print("==========================================")
print("VISUAL COMPARISON CREATED")
print("==========================================")
print("Image :", filename)
print("Saved :", output_path)
print("==========================================")