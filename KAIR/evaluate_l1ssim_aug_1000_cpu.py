import os
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from models.network_swinir import SwinIR

# ============================================================
# PATHS
# ============================================================

checkpoint = r"D:\KLA_SwinIR_Project\results\l1_ssim_aug_cpu_1000\l1_ssim_aug_1000iter_G.pth"

noisy_dir = r"D:\KLA_SwinIR_Project\datasets\validation\NoisyLR"
gt_dir = r"D:\KLA_SwinIR_Project\datasets\validation\GT"

output_dir = r"D:\KLA_SwinIR_Project\results\validation_l1ssim_aug_1000_cpu"

os.makedirs(output_dir, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("==========================================")
print("FULL SWINIR VALIDATION")
print("==========================================")
print("Device:", device)
print("Checkpoint:", checkpoint)
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
)

checkpoint_data = torch.load(
    checkpoint,
    map_location="cpu"
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
model = model.to(device)
model.eval()

print("100000_E.pth loaded successfully.")
print("Parameters:", sum(p.numel() for p in model.parameters()))
print()

# ============================================================
# GET MATCHING FILES
# ============================================================

noisy_files = sorted(
    f for f in os.listdir(noisy_dir)
    if f.lower().endswith(".npy")
)

gt_files = set(
    f for f in os.listdir(gt_dir)
    if f.lower().endswith(".npy")
)

pairs = [
    f for f in noisy_files
    if f in gt_files
]

print("Validation pairs:", len(pairs))
print()

# ============================================================
# METRIC STORAGE
# ============================================================

psnr_values = []
ssim_values = []

# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for i, filename in enumerate(pairs, start=1):

    noisy_path = os.path.join(noisy_dir, filename)
    gt_path = os.path.join(gt_dir, filename)

    noisy = np.load(noisy_path).astype(np.float32)
    gt = np.load(gt_path).astype(np.float32)

    noisy = np.squeeze(noisy)
    gt = np.squeeze(gt)

    if noisy.shape != (128, 128):
        print("Skipping", filename, "invalid noisy shape:", noisy.shape)
        continue

    if gt.shape != (256, 256):
        print("Skipping", filename, "invalid GT shape:", gt.shape)
        continue

    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    noisy_tensor = torch.from_numpy(
        noisy
    ).unsqueeze(0).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():
        restored = model(noisy_tensor)

    restored = restored.squeeze().cpu().numpy()

    if restored.shape != gt.shape:
        print(
            "Skipping",
            filename,
            "output shape:",
            restored.shape
        )
        continue

    # --------------------------------------------------------
    # Clip output
    # --------------------------------------------------------

    restored = np.clip(restored, 0.0, 1.0)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    psnr = peak_signal_noise_ratio(
        gt,
        restored,
        data_range=1.0
    )

    ssim = structural_similarity(
        gt,
        restored,
        data_range=1.0
    )

    psnr_values.append(psnr)
    ssim_values.append(ssim)

    # --------------------------------------------------------
    # Save restored image
    # --------------------------------------------------------

    output_path = os.path.join(
        output_dir,
        filename
    )

    np.save(output_path, restored)

    print(
        f"[{i}/{len(pairs)}] "
        f"{filename} | "
        f"PSNR: {psnr:.4f} dB | "
        f"SSIM: {ssim:.4f}"
    )

# ============================================================
# FINAL RESULTS
# ============================================================

if not psnr_values:
    raise RuntimeError("No validation images were successfully evaluated.")

average_psnr = np.mean(psnr_values)
average_ssim = np.mean(ssim_values)

best_psnr = np.max(psnr_values)
worst_psnr = np.min(psnr_values)

best_ssim = np.max(ssim_values)
worst_ssim = np.min(ssim_values)

print()
print("==========================================")
print("       FINAL VALIDATION RESULTS")
print("==========================================")
print("Images evaluated :", len(psnr_values))
print()
print(f"Average PSNR     : {average_psnr:.4f} dB")
print(f"Average SSIM     : {average_ssim:.4f}")
print()
print(f"Best PSNR        : {best_psnr:.4f} dB")
print(f"Worst PSNR       : {worst_psnr:.4f} dB")
print()
print(f"Best SSIM        : {best_ssim:.4f}")
print(f"Worst SSIM       : {worst_ssim:.4f}")
print("==========================================")
print()
print("Restored images saved in:")
print(output_dir)





