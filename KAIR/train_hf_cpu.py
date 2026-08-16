import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# ============================================================
# KAIR IMPORT PATH
# ============================================================

KAIR_DIR = os.path.dirname(os.path.abspath(__file__))

if KAIR_DIR not in sys.path:
    sys.path.insert(0, KAIR_DIR)

from models.network_swinir import SwinIR
from models.loss_ssim import SSIMLoss


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(KAIR_DIR)

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "results",
    "l1_ssim_aug_cpu_1000",
    "l1_ssim_aug_1000iter_G.pth"
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
    "hf_loss_cpu_500"
)

FINAL_CHECKPOINT = os.path.join(
    OUTPUT_DIR,
    "l1_ssim_aug_hf_500iter_G.pth"
)

LOSS_HISTORY = os.path.join(
    OUTPUT_DIR,
    "loss_history.txt"
)


# ============================================================
# TRAINING CONFIG
# ============================================================

DEVICE = torch.device("cpu")

MAX_ITERS = 500

LEARNING_RATE = 1e-5

SSIM_WEIGHT = 0.1

HF_WEIGHT = 0.05

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# PATH CHECKS
# ============================================================

if not os.path.isfile(CHECKPOINT):
    raise FileNotFoundError(
        f"Starting checkpoint not found:\n{CHECKPOINT}"
    )

if not os.path.isdir(TRAIN_LR_DIR):
    raise FileNotFoundError(
        f"NoisyLR directory not found:\n{TRAIN_LR_DIR}"
    )

if not os.path.isdir(TRAIN_GT_DIR):
    raise FileNotFoundError(
        f"GT directory not found:\n{TRAIN_GT_DIR}"
    )

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DATA PAIRS
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
    raise RuntimeError(
        "No matching training pairs found."
    )


# ============================================================
# SAME GEOMETRIC AUGMENTATION FOR LQ + GT
# ============================================================

def augment_pair(noisy, gt):

    if random.random() < 0.5:
        noisy = np.fliplr(noisy)
        gt = np.fliplr(gt)

    if random.random() < 0.5:
        noisy = np.flipud(noisy)
        gt = np.flipud(gt)

    k = random.randint(0, 3)

    if k != 0:
        noisy = np.rot90(noisy, k=k)
        gt = np.rot90(gt, k=k)

    noisy = np.ascontiguousarray(noisy)
    gt = np.ascontiguousarray(gt)

    return noisy, gt


# ============================================================
# HIGH-FREQUENCY / GRADIENT LOSS
# ============================================================

def gradient_loss(output, target):

    output_x = output[:, :, :, 1:] - output[:, :, :, :-1]
    target_x = target[:, :, :, 1:] - target[:, :, :, :-1]

    output_y = output[:, :, 1:, :] - output[:, :, :-1, :]
    target_y = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = F.l1_loss(
        output_x,
        target_x
    )

    loss_y = F.l1_loss(
        output_y,
        target_y
    )

    return loss_x + loss_y


# ============================================================
# HEADER
# ============================================================

print("==========================================")
print(" L1 + SSIM + HF LOSS CPU TRAINING")
print("==========================================")
print("Device        :", DEVICE)
print("Checkpoint    :", CHECKPOINT)
print("Training pairs:", len(pairs))
print("Iterations    :", MAX_ITERS)
print("Learning rate :", LEARNING_RATE)
print("SSIM weight   :", SSIM_WEIGHT)
print("HF weight     :", HF_WEIGHT)
print()
print(
    "Loss = L1 + 0.1*(1-SSIM) + 0.05*HF"
)
print("==========================================")
print()


# ============================================================
# MODEL
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
# LOAD PREVIOUS BEST CANDIDATE
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


model.load_state_dict(
    state_dict,
    strict=True
)

print("Starting checkpoint loaded successfully.")

print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)

print()


# ============================================================
# LOSSES
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
# TRAIN
# ============================================================

model.train()

loss_history = []

for iteration in range(
    1,
    MAX_ITERS + 1
):

    filename = random.choice(pairs)

    noisy = np.load(
        os.path.join(
            TRAIN_LR_DIR,
            filename
        )
    ).astype(np.float32)

    gt = np.load(
        os.path.join(
            TRAIN_GT_DIR,
            filename
        )
    ).astype(np.float32)

    noisy = np.squeeze(noisy)
    gt = np.squeeze(gt)

    if noisy.shape != (128, 128):
        raise ValueError(
            f"{filename}: invalid NoisyLR shape "
            f"{noisy.shape}"
        )

    if gt.shape != (256, 256):
        raise ValueError(
            f"{filename}: invalid GT shape "
            f"{gt.shape}"
        )

    # Same geometric transform
    noisy, gt = augment_pair(
        noisy,
        gt
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

    optimizer.zero_grad()

    restored = model(
        noisy_tensor
    )

    # L1
    l1_value = l1_loss_fn(
        restored,
        gt_tensor
    )

    # SSIM
    ssim_value = ssim_loss_fn(
        restored,
        gt_tensor
    )

    # High-frequency / edge term
    hf_value = gradient_loss(
        restored,
        gt_tensor
    )

    # Combined objective
    combined_loss = (
        l1_value
        + SSIM_WEIGHT * (
            1.0 - ssim_value
        )
        + HF_WEIGHT * hf_value
    )

    combined_loss.backward()

    optimizer.step()

    loss_history.append(
        combined_loss.item()
    )

    if (
        iteration == 1
        or iteration % 50 == 0
    ):

        print(
            f"Iter {iteration:3d}/{MAX_ITERS} | "
            f"L1: {l1_value.item():.6f} | "
            f"SSIM: {ssim_value.item():.6f} | "
            f"HF: {hf_value.item():.6f} | "
            f"Combined: {combined_loss.item():.6f}"
        )


# ============================================================
# SAVE
# ============================================================

torch.save(
    model.state_dict(),
    FINAL_CHECKPOINT
)

with open(
    LOSS_HISTORY,
    "w"
) as f:

    for i, value in enumerate(
        loss_history,
        start=1
    ):

        f.write(
            f"{i},{value:.10f}\n"
        )


print()
print("==========================================")
print(" HF LOSS EXPERIMENT FINISHED")
print("==========================================")
print(
    "Iterations completed :",
    MAX_ITERS
)
print(
    "Final combined loss  :",
    f"{loss_history[-1]:.8f}"
)
print()
print("Checkpoint:")
print(FINAL_CHECKPOINT)
print()
print("Loss history:")
print(LOSS_HISTORY)
print("==========================================")