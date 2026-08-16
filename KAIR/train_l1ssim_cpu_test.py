import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# ------------------------------------------------------------
# Make KAIR imports work
# ------------------------------------------------------------
KAIR_DIR = os.path.dirname(os.path.abspath(__file__))
if KAIR_DIR not in sys.path:
    sys.path.insert(0, KAIR_DIR)

from models.network_swinir import SwinIR
from models.loss_ssim import SSIMLoss


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(KAIR_DIR)

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "swinir_kla",
    "swinir_kla_x2",
    "models",
    "100000_E.pth"
)

TRAIN_LR_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "train",
    "NoisyLR"
)

TRAIN_GT_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "train",
    "GT"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "l1_ssim_cpu_test"
)

MAX_ITERS = 100
LEARNING_RATE = 0.0001
SSIM_WEIGHT = 0.1

DEVICE = torch.device("cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.isfile(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

if not os.path.isdir(TRAIN_LR_DIR):
    raise FileNotFoundError(
        f"Training NoisyLR folder not found:\n{TRAIN_LR_DIR}"
    )

if not os.path.isdir(TRAIN_GT_DIR):
    raise FileNotFoundError(
        f"Training GT folder not found:\n{TRAIN_GT_DIR}"
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIND PAIRED TRAINING FILES
# ============================================================

lr_files = sorted(
    f for f in os.listdir(TRAIN_LR_DIR)
    if f.lower().endswith(".npy")
)

gt_files = {
    f for f in os.listdir(TRAIN_GT_DIR)
    if f.lower().endswith(".npy")
}

pairs = [
    f for f in lr_files
    if f in gt_files
]

if not pairs:
    raise RuntimeError("No matching training pairs found.")

print("==========================================")
print("      L1 + SSIM CPU SANITY TRAINING")
print("==========================================")
print("Device           :", DEVICE)
print("Checkpoint       :", CHECKPOINT)
print("Training pairs   :", len(pairs))
print("Iterations       :", MAX_ITERS)
print("Learning rate    :", LEARNING_RATE)
print("SSIM weight      :", SSIM_WEIGHT)
print("Loss             : L1 + 0.1 * (1 - SSIM)")
print("==========================================")
print()


# ============================================================
# BUILD EXACT SWINIR ARCHITECTURE
# ============================================================

model = SwinIR(
    upscale=2,
    in_chans=1,
    img_size=128,
    window_size=8,
    img_range=1.0,
    depths=[2, 2, 2, 2],
    embed_dim=60,
    num_heads=[3, 3, 3, 3],
    mlp_ratio=2,
    upsampler="pixelshuffle",
    resi_connection="1conv"
).to(DEVICE)


# ============================================================
# LOAD 100K CHECKPOINT
# ============================================================

checkpoint_data = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

if isinstance(checkpoint_data, dict):

    if "params_ema" in checkpoint_data:
        state_dict = checkpoint_data["params_ema"]

    elif "params" in checkpoint_data:
        state_dict = checkpoint_data["params"]

    else:
        state_dict = checkpoint_data

else:
    state_dict = checkpoint_data


model.load_state_dict(state_dict, strict=True)

print(
    "100000_E.pth loaded successfully."
)

print(
    "Parameters        :",
    sum(p.numel() for p in model.parameters())
)

print()


# ============================================================
# LOSS FUNCTIONS
# ============================================================

l1_loss_fn = nn.L1Loss()

ssim_loss_fn = SSIMLoss(
    window_size=11,
    size_average=True
).to(DEVICE)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

model.train()

loss_history = []

for iteration in range(1, MAX_ITERS + 1):

    # --------------------------------------------------------
    # Random training pair
    # --------------------------------------------------------

    filename = random.choice(pairs)

    lr_path = os.path.join(
        TRAIN_LR_DIR,
        filename
    )

    gt_path = os.path.join(
        TRAIN_GT_DIR,
        filename
    )

    noisy = np.load(lr_path).astype(np.float32)
    gt = np.load(gt_path).astype(np.float32)

    noisy = np.squeeze(noisy)
    gt = np.squeeze(gt)

    if noisy.shape != (128, 128):
        raise ValueError(
            f"Invalid NoisyLR shape for {filename}: "
            f"{noisy.shape}"
        )

    if gt.shape != (256, 256):
        raise ValueError(
            f"Invalid GT shape for {filename}: "
            f"{gt.shape}"
        )

    noisy_tensor = (
        torch.from_numpy(noisy)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    gt_tensor = (
        torch.from_numpy(gt)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    optimizer.zero_grad()

    restored = model(noisy_tensor)

    # --------------------------------------------------------
    # L1
    # --------------------------------------------------------

    l1_value = l1_loss_fn(
        restored,
        gt_tensor
    )

    # --------------------------------------------------------
    # SSIM
    # SSIMLoss returns similarity itself.
    # Therefore we minimize (1 - SSIM).
    # --------------------------------------------------------

    ssim_value = ssim_loss_fn(
        restored,
        gt_tensor
    )

    combined_loss = (
        l1_value
        + SSIM_WEIGHT * (1.0 - ssim_value)
    )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    combined_loss.backward()

    optimizer.step()

    loss_history.append(
        combined_loss.item()
    )

    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    if iteration == 1 or iteration % 10 == 0:

        print(
            f"Iter {iteration:3d}/{MAX_ITERS} | "
            f"L1: {l1_value.item():.6f} | "
            f"SSIM: {ssim_value.item():.6f} | "
            f"Combined: {combined_loss.item():.6f}"
        )


# ============================================================
# SAVE SANITY CHECKPOINT
# ============================================================

sanity_checkpoint = os.path.join(
    OUTPUT_DIR,
    "l1_ssim_cpu_100iter_G.pth"
)

torch.save(
    model.state_dict(),
    sanity_checkpoint
)

# ------------------------------------------------------------
# Save loss history
# ------------------------------------------------------------

loss_history_path = os.path.join(
    OUTPUT_DIR,
    "loss_history.txt"
)

with open(loss_history_path, "w") as f:

    for i, value in enumerate(
        loss_history,
        start=1
    ):

        f.write(
            f"{i},{value:.10f}\n"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("==========================================")
print("     L1 + SSIM SANITY TEST COMPLETED")
print("==========================================")
print("Iterations completed :", MAX_ITERS)
print("Final combined loss  :", f"{loss_history[-1]:.8f}")
print()
print("Checkpoint saved to:")
print(sanity_checkpoint)
print()
print("Loss history saved to:")
print(loss_history_path)
print("==========================================")