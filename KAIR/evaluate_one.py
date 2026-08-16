import os
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from models.network_swinir import SwinIR


# ============================================================
# CONFIGURATION
# ============================================================

checkpoint = "../swinir_kla/swinir_kla_x2/models/65000_E.pth"

noisy_path = "../datasets/validation/NoisyLR/000002.npy"
gt_path = "../datasets/validation/GT/000002.npy"

output_path = "../results/000002_restored_E.npy"


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("==========================================")
print("SwinIR Evaluation")
print("==========================================")
print("Device:", device)
print("Checkpoint:", checkpoint)
print()


# ============================================================
# BUILD THE EXACT LIGHTWEIGHT SWINIR MODEL
# USED DURING TRAINING
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
)


print("Model created.")
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# ============================================================
# LOAD EMA CHECKPOINT
# ============================================================

if not os.path.exists(checkpoint):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{checkpoint}"
    )


checkpoint_data = torch.load(
    checkpoint,
    map_location="cpu"
)


print()
print("Checkpoint loaded from disk.")
print("Checkpoint type:", type(checkpoint_data))


# KAIR checkpoints normally contain "params"
# or another state-dict-like structure.

if isinstance(checkpoint_data, dict):

    if "params_ema" in checkpoint_data:
        state_dict = checkpoint_data["params_ema"]
        print("Using: params_ema")

    elif "params" in checkpoint_data:
        state_dict = checkpoint_data["params"]
        print("Using: params")

    else:
        state_dict = checkpoint_data
        print("Using checkpoint dictionary directly.")

else:
    state_dict = checkpoint_data
    print("Using checkpoint directly.")


# ============================================================
# LOAD WEIGHTS
# ============================================================

model.load_state_dict(
    state_dict,
    strict=True
)

model = model.to(device)
model.eval()

print()
print("65000_E.pth loaded successfully.")
print()


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

if not os.path.exists(noisy_path):
    raise FileNotFoundError(
        f"Noisy image not found:\n{noisy_path}"
    )

if not os.path.exists(gt_path):
    raise FileNotFoundError(
        f"GT image not found:\n{gt_path}"
    )


noisy = np.load(noisy_path).astype(np.float32)
gt = np.load(gt_path).astype(np.float32)


print("Noisy original shape:", noisy.shape)
print("GT original shape   :", gt.shape)

print()
print("Noisy statistics:")
print("  min :", noisy.min())
print("  max :", noisy.max())
print("  mean:", noisy.mean())

print()
print("GT statistics:")
print("  min :", gt.min())
print("  max :", gt.max())
print("  mean:", gt.mean())


# ============================================================
# IMPORTANT:
# DO NOT NORMALIZE THE DATA
#
# Your DatasetKLA training loader directly loads:
#
# np.load(...).astype(np.float32)
#
# Therefore evaluation uses the same representation.
# ============================================================

noisy = np.squeeze(noisy)
gt = np.squeeze(gt)


print()
print("Noisy shape after squeeze:", noisy.shape)
print("GT shape after squeeze   :", gt.shape)


# ============================================================
# VERIFY EXPECTED INPUT / TARGET SIZE
# ============================================================

if noisy.shape != (128, 128):
    raise ValueError(
        f"Expected noisy image shape (128,128), "
        f"but got {noisy.shape}"
    )

if gt.shape != (256, 256):
    raise ValueError(
        f"Expected GT shape (256,256), "
        f"but got {gt.shape}"
    )


# ============================================================
# CONVERT TO TORCH
#
# NumPy:
#       H x W
#
# becomes:
#       1 x 1 x H x W
# ============================================================

noisy_tensor = torch.from_numpy(
    noisy
).unsqueeze(0).unsqueeze(0).to(device)


print()
print("Input tensor shape:", noisy_tensor.shape)
print("Input tensor dtype:", noisy_tensor.dtype)


# ============================================================
# INFERENCE
# ============================================================

print()
print("Starting inference...")

with torch.no_grad():

    restored = model(noisy_tensor)


# ============================================================
# CONVERT OUTPUT TO NUMPY
# ============================================================

restored = restored.squeeze().cpu().numpy()


print("Restored shape:", restored.shape)

print()
print("Raw model output statistics:")
print("  min :", restored.min())
print("  max :", restored.max())
print("  mean:", restored.mean())


# ============================================================
# CHECK OUTPUT SIZE
# ============================================================

if restored.shape != gt.shape:

    raise ValueError(
        f"Model output shape {restored.shape} "
        f"does not match GT shape {gt.shape}"
    )


# ============================================================
# CLIP OUTPUT
#
# GT is in [0,1], so constrain the final restored image
# to the same range before calculating PSNR/SSIM.
# ============================================================

restored = np.clip(
    restored,
    0.0,
    1.0
)


print()
print("Clipped output statistics:")
print("  min :", restored.min())
print("  max :", restored.max())
print("  mean:", restored.mean())


# ============================================================
# PSNR
# ============================================================

print()
print("Calculating PSNR...")

psnr = peak_signal_noise_ratio(
    gt,
    restored,
    data_range=1.0
)


# ============================================================
# SSIM
# ============================================================

print("Calculating SSIM...")

ssim = structural_similarity(
    gt,
    restored,
    data_range=1.0
)


# ============================================================
# RESULTS
# ============================================================

print()
print("==========================================")
print("         EVALUATION RESULT")
print("==========================================")

print("Image       : 000002.npy")
print("Checkpoint  : 65000_E.pth")

print()
print("Input shape :", noisy.shape)
print("GT shape    :", gt.shape)
print("Output shape:", restored.shape)

print()
print("PSNR        :", psnr, "dB")
print("SSIM        :", ssim)

print("==========================================")


# ============================================================
# SAVE RESTORED IMAGE
# ============================================================

os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)

np.save(
    output_path,
    restored
)


print()
print("Restored image saved to:")
print(output_path)