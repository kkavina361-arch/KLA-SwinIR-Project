import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CASES
# ============================================================

CASES = [
    "000108.npy",
    "000008.npy",
    "000086.npy",
]


# ============================================================
# PATHS
# ============================================================

NOISY_DIR = r"D:\KLA_SwinIR_Project\datasets\validation\NoisyLR"

GT_DIR = r"D:\KLA_SwinIR_Project\datasets\validation\GT"

BASELINE_DIR = r"D:\results\validation_100000_E_cpu"

PROPOSED_DIR = (
    r"D:\KLA_SwinIR_Project"
    r"\results\validation_l1ssim_aug_1000_cpu"
)

OUTPUT_DIR = (
    r"D:\KLA_SwinIR_Project"
    r"\results\final_visual_comparisons"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CHECK DIRECTORIES
# ============================================================

required_dirs = [
    NOISY_DIR,
    GT_DIR,
    BASELINE_DIR,
    PROPOSED_DIR,
]

for folder in required_dirs:

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Required directory not found:\n{folder}"
        )


# ============================================================
# PROCESS CASES
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

    proposed_path = os.path.join(
        PROPOSED_DIR,
        filename
    )

    gt_path = os.path.join(
        GT_DIR,
        filename
    )


    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    for path in [
        noisy_path,
        baseline_path,
        proposed_path,
        gt_path,
    ]:

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    noisy = np.load(
        noisy_path
    ).astype(np.float32)

    baseline = np.load(
        baseline_path
    ).astype(np.float32)

    proposed = np.load(
        proposed_path
    ).astype(np.float32)

    gt = np.load(
        gt_path
    ).astype(np.float32)


    # --------------------------------------------------------
    # Squeeze
    # --------------------------------------------------------

    noisy = np.squeeze(noisy)
    baseline = np.squeeze(baseline)
    proposed = np.squeeze(proposed)
    gt = np.squeeze(gt)


    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    if noisy.shape != (128, 128):
        raise ValueError(
            f"{filename}: NoisyLR shape is {noisy.shape}, "
            f"expected (128,128)"
        )

    for name, image in [
        ("Baseline", baseline),
        ("Proposed", proposed),
        ("GT", gt),
    ]:

        if image.shape != (256, 256):
            raise ValueError(
                f"{filename}: {name} shape is {image.shape}, "
                f"expected (256,256)"
            )


    # --------------------------------------------------------
    # Clip display values
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

    proposed = np.clip(
        proposed,
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
    # Degraded
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
    # Baseline
    # --------------------------------------------------------

    axes[1].imshow(
        baseline,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        "100K L1 Baseline\n256 × 256",
        fontsize=12
    )

    axes[1].axis("off")


    # --------------------------------------------------------
    # Proposed
    # --------------------------------------------------------

    axes[2].imshow(
        proposed,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        "L1 + SSIM + Aug.\n1000 Iterations",
        fontsize=12
    )

    axes[2].axis("off")


    # --------------------------------------------------------
    # Ground Truth
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
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        f"KLA SwinIR Restoration — {filename}",
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
        f"FINAL_{filename[:-4]}.png"
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
# COMPLETE
# ============================================================

print()
print("==========================================")
print(" FINAL VISUAL COMPARISONS CREATED")
print("==========================================")
print("Cases:")
print("  000108")
print("  000008")
print("  000086")
print()
print("Saved in:")
print(OUTPUT_DIR)
print("==========================================")