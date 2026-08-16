import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ============================================================
# PATHS
# ============================================================

noisy_dir = "../datasets/validation/NoisyLR"
gt_dir = "../datasets/validation/GT"
output_dir = "../results/validation_65000_E"

save_dir = "../results/sir_comparison"
os.makedirs(save_dir, exist_ok=True)

# ============================================================
# FIND VALIDATION FILES
# ============================================================

noisy_files = {
    f for f in os.listdir(noisy_dir)
    if f.endswith(".npy")
}

gt_files = {
    f for f in os.listdir(gt_dir)
    if f.endswith(".npy")
}

output_files = {
    f for f in os.listdir(output_dir)
    if f.endswith(".npy")
}

files = sorted(noisy_files & gt_files & output_files)

print("Validation images found:", len(files))

# ============================================================
# CALCULATE METRICS FOR ALL IMAGES
# ============================================================

results = []

for filename in files:

    gt = np.load(
        os.path.join(gt_dir, filename)
    ).astype(np.float32)

    output = np.load(
        os.path.join(output_dir, filename)
    ).astype(np.float32)

    gt = np.squeeze(gt)
    output = np.squeeze(output)

    if gt.shape != output.shape:
        continue

    output = np.clip(output, 0.0, 1.0)

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

    results.append(
        (filename, psnr, ssim)
    )

print("Successfully evaluated:", len(results))

# ============================================================
# SELECT BEST / REPRESENTATIVE / WORST
# ============================================================

results.sort(key=lambda x: x[1])

worst = results[0]
best = results[-1]

target_psnr = 28.3543

representative = min(
    results,
    key=lambda x: abs(x[1] - target_psnr)
)

selected = [
    ("BEST CASE", best),
    ("REPRESENTATIVE CASE", representative),
    ("DIFFICULT CASE", worst)
]

print()
print("Selected images:")
print(
    "Best          :",
    best[0],
    f"PSNR={best[1]:.4f}",
    f"SSIM={best[2]:.4f}"
)
print(
    "Representative:",
    representative[0],
    f"PSNR={representative[1]:.4f}",
    f"SSIM={representative[2]:.4f}"
)
print(
    "Difficult     :",
    worst[0],
    f"PSNR={worst[1]:.4f}",
    f"SSIM={worst[2]:.4f}"
)

# ============================================================
# CREATE INDIVIDUAL FIGURES
# ============================================================

for label, (filename, psnr, ssim) in selected:

    noisy = np.load(
        os.path.join(noisy_dir, filename)
    ).astype(np.float32)

    gt = np.load(
        os.path.join(gt_dir, filename)
    ).astype(np.float32)

    output = np.load(
        os.path.join(output_dir, filename)
    ).astype(np.float32)

    noisy = np.squeeze(noisy)
    gt = np.squeeze(gt)
    output = np.squeeze(output)

    # Upscale noisy image ONLY for visualization.
    noisy_display = resize(
        noisy,
        gt.shape,
        order=1,
        preserve_range=True,
        anti_aliasing=False
    )

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
        f"SwinIR Output\n"
        f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.3f}"
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

    fig.suptitle(
        f"{label} — {filename}",
        fontsize=15
    )

    plt.tight_layout()

    safe_name = label.lower().replace(
        " ",
        "_"
    )

    save_path = os.path.join(
        save_dir,
        f"{safe_name}_{filename.replace('.npy', '.png')}"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# CREATE ONE COMBINED REPORT FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    3,
    figsize=(13, 13)
)

for row, (label, (filename, psnr, ssim)) in enumerate(selected):

    noisy = np.squeeze(
        np.load(
            os.path.join(noisy_dir, filename)
        ).astype(np.float32)
    )

    gt = np.squeeze(
        np.load(
            os.path.join(gt_dir, filename)
        ).astype(np.float32)
    )

    output = np.squeeze(
        np.load(
            os.path.join(output_dir, filename)
        ).astype(np.float32)
    )

    noisy_display = resize(
        noisy,
        gt.shape,
        order=1,
        preserve_range=True,
        anti_aliasing=False
    )

    axes[row, 0].imshow(
        np.clip(noisy_display, 0, 1),
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[row, 0].set_title(
        f"{label}\nNoisy Input"
    )

    axes[row, 1].imshow(
        np.clip(output, 0, 1),
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[row, 1].set_title(
        f"SwinIR Output\n"
        f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.3f}"
    )

    axes[row, 2].imshow(
        np.clip(gt, 0, 1),
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[row, 2].set_title(
        "Ground Truth"
    )

    for col in range(3):
        axes[row, col].axis("off")

fig.suptitle(
    "SwinIR Visual Comparison — 65,000 Iterations",
    fontsize=17
)

plt.tight_layout(
    rect=[0, 0, 1, 0.97]
)

combined_path = os.path.join(
    save_dir,
    "SwinIR_visual_comparison_65000.png"
)

plt.savefig(
    combined_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# SUMMARY FILE
# ============================================================

summary_path = os.path.join(
    save_dir,
    "comparison_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SwinIR Visual Comparison - 65,000 Iterations\n"
    )

    f.write(
        "Validation images: 320\n"
    )

    f.write(
        "Average PSNR: 28.3543 dB\n"
    )

    f.write(
        "Average SSIM: 0.7594\n\n"
    )

    for label, (filename, psnr, ssim) in selected:

        f.write(
            f"{label}: {filename} | "
            f"PSNR={psnr:.4f} dB | "
            f"SSIM={ssim:.4f}\n"
        )

print()
print("==========================================")
print("COMPARISON PACKAGE CREATED")
print("==========================================")
print("Folder:")
print(os.path.abspath(save_dir))
print()
print("Combined figure:")
print(os.path.abspath(combined_path))
print()
print("Summary:")
print(os.path.abspath(summary_path))
print("==========================================")