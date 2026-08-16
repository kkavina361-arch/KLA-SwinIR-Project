"""
Project : AI-Based Restoration of Degraded Semiconductor Images
Model   : SwinIR
Dataset : KLA Semiconductor Dataset
Author  : Your Team
"""

import os
import random
import shutil
from pathlib import Path

# =====================================================
# Configuration
# =====================================================

SEED = 42
TRAIN_RATIO = 0.90

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_GT = PROJECT_ROOT / "datasets" / "original_train" / "GT"
ORIGINAL_LR = PROJECT_ROOT / "datasets" / "original_train" / "NoisyLR"

TRAIN_GT = PROJECT_ROOT / "datasets" / "train" / "GT"
TRAIN_LR = PROJECT_ROOT / "datasets" / "train" / "NoisyLR"

VAL_GT = PROJECT_ROOT / "datasets" / "validation" / "GT"
VAL_LR = PROJECT_ROOT / "datasets" / "validation" / "NoisyLR"

# =====================================================
# Create folders
# =====================================================

TRAIN_GT.mkdir(parents=True, exist_ok=True)
TRAIN_LR.mkdir(parents=True, exist_ok=True)

VAL_GT.mkdir(parents=True, exist_ok=True)
VAL_LR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Read filenames
# =====================================================

gt_files = sorted([f for f in os.listdir(ORIGINAL_GT) if f.endswith(".npy")])
lr_files = sorted([f for f in os.listdir(ORIGINAL_LR) if f.endswith(".npy")])

# =====================================================
# Verify dataset integrity
# =====================================================

if len(gt_files) != len(lr_files):
    raise ValueError("Mismatch between GT and NoisyLR image counts.")

for file in gt_files:
    if file not in lr_files:
        raise ValueError(f"Missing matching NoisyLR file: {file}")

print("\nDataset verification successful.")
print(f"Total image pairs : {len(gt_files)}")

# =====================================================
# Shuffle
# =====================================================

random.seed(SEED)

pairs = gt_files.copy()
random.shuffle(pairs)

split_index = int(len(pairs) * TRAIN_RATIO)

train_pairs = pairs[:split_index]
val_pairs = pairs[split_index:]

# =====================================================
# Copy Training Files
# =====================================================

print("\nCopying training images...")

for file in train_pairs:

    shutil.copy2(
        ORIGINAL_GT / file,
        TRAIN_GT / file
    )

    shutil.copy2(
        ORIGINAL_LR / file,
        TRAIN_LR / file
    )

# =====================================================
# Copy Validation Files
# =====================================================

print("Copying validation images...")

for file in val_pairs:

    shutil.copy2(
        ORIGINAL_GT / file,
        VAL_GT / file
    )

    shutil.copy2(
        ORIGINAL_LR / file,
        VAL_LR / file
    )

# =====================================================
# Summary
# =====================================================

print("\n===========================================")
print("Dataset Split Completed Successfully")
print("===========================================")

print(f"Total Images       : {len(pairs)}")
print(f"Training Images    : {len(train_pairs)}")
print(f"Validation Images  : {len(val_pairs)}")

print("\nFolders Created:")

print(TRAIN_GT)
print(TRAIN_LR)
print(VAL_GT)
print(VAL_LR)

print("\nRandom Seed :", SEED)

print("\nReady for SwinIR Training!")