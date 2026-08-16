import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CASES TO COMPARE
# ============================================================

CASES = [
    "000108.npy",
    "000008.npy",
    "000086.npy",
]


# ============================================================
# DIRECTORIES
# ============================================================

NOISY_DIR = r"D:\KLA_SwinIR_Project\datasets\validation\NoisyLR"

GT_DIR = r"D:\KLA_SwinIR_Project\datasets\validation\GT"

BASELINE_DIR = r"D:\results\validation_100000_E_cpu"

L1SSIM_DIR = r"D:\KLA_SwinIR_Project\results\validation_l1ssim_100_cpu"

OUTPUT_DIR = r"D:\KLA_SwinIR_Project\results\final_visual_comparisons"


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CHECK REQUIRED DIRECTORIES
# ============================================================

for folder in [
    NOISY_DIR,
    GT_DIR,
    BASELINE_DIR,
    L1SSIM_DIR,
]:

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Required folder not found:\n{folder}"
        )


# ============================================================
# PROCESS EACH CASE
# ============================================================

for filename in CASES:

    print()
    print("Processing:", filename)

    noisy_path = os.path.join(
        NOISY_DIR,
        filename
    )

    baseline_path = os.path.join(
        BASELINE_DIR,
        filename
    )

    l1ssim_path = os.path.join(
        L1SSIM_DIR,
        filename
    )

    gt_path = os.path.join(
        GT_DIR,
        filename
    )


    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        noisy_path,
        baseline_path,
        l1ssim_path,
        gt_path,
    ]:

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )


    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    noisy = np.load(noisy_path).astype(np.float32)

    baseline = np.load(
        baseline_path
    ).astype(np.float32)

    l1ssim = np.load(
        l1ssim_path
    ).astype(np.float32)

    gt = np.load(gt_path).astype(np.float32)


    # --------------------------------------------------------
    # Remove unnecessary dimensions
    # --------------------------------------------------------

    noisy = np.squeeze(noisy)

    baseline = np.squeeze(baseline)

    l1ssim = np.squeeze(l1ssim)

    gt = np.squeeze(gt)


    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    if noisy.shape != (128, 128):
        raise ValueError(
            f"{filename}: expected NoisyLR "
            f"(128,128), got {noisy.shape}"
        )

    if baseline.shape != (256, 256):
        raise ValueError(
            f"{filename}: expected baseline "
            f"(256,256), got {baseline.shape}"
        )

    if l1ssim.shape != (256, 256):
        raise ValueError(
            f"{filename}: expected L1+SSIM "
            f"(256,256), got {l1ssim.shape}"
        )

    if gt.shape != (256, 256):
        raise ValueError(
            f"{filename}: expected GT "
            f"(256,256), got {gt.shape}"
        )


    # --------------------------------------------------------
    # Clip for display
    # --------------------------------------------------------

    noisy = np.clip(
        noisy,
        0.0,
        1.0
    )

    baseline = np.clip(
        baseline,
        0.0,
        1.0
    )

    l1ssim = np.clip(
        l1ssim,
        0.0,
        1.0
    )

    gt = np.clip(
        gt,
        0.0,
        1.0
    )


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )


    # --------------------------------------------------------
    # 1. Degraded input
    # --------------------------------------------------------

    axes[0].imshow(
        noisy,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0].set_title(
        "Degraded Input\n128 × 128",
        fontsize=12
    )

    axes[0].axis("off")


    # --------------------------------------------------------
    # 2. 100K L1 baseline
    # --------------------------------------------------------

    axes[1].imshow(
        baseline,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        "100K L1\n256 × 256",
        fontsize=12
    )

    axes[1].axis("off")


    # --------------------------------------------------------
    # 3. L1 + SSIM
    # --------------------------------------------------------

    axes[2].imshow(
        l1ssim,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        "L1 + SSIM\n100 iterations",
        fontsize=12
    )

    axes[2].axis("off")


    # --------------------------------------------------------
    # 4. Ground Truth
    # --------------------------------------------------------

    axes[3].imshow(
        gt,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[3].set_title(
        "Ground Truth\n256 × 256",
        fontsize=12
    )

    axes[3].axis("off")


    # --------------------------------------------------------
    # Figure title
    # --------------------------------------------------------

    fig.suptitle(
        f"KLA SwinIR Restoration Comparison — {filename}",
        fontsize=14
    )


    plt.tight_layout(
        rect=[0, 0, 1, 0.94]
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"comparison_{filename[:-4]}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "Created:",
        output_path
    )


# ============================================================
# DONE
# ============================================================

print()
print("==========================================")
print("THREE VISUAL COMPARISONS CREATED")
print("==========================================")
print("Cases:")
print("  000108")
print("  000008")
print("  000086")
print()
print("Output folder:")
print(OUTPUT_DIR)
print("==========================================")