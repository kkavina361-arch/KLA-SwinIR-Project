import argparse
import os
import time

import numpy as np
import torch

from KAIR.models.network_swinir import SwinIR


# ============================================================
# SWINIR MODEL
# ============================================================

def build_model(model_path, device):

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
        model_path,
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

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# SAVE PNG
# ============================================================

def save_png(array, path):

    import cv2

    image_uint8 = (
        np.clip(array, 0.0, 1.0) * 255.0
    ).round().astype(np.uint8)

    cv2.imwrite(
        path,
        image_uint8
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Standalone KLA SwinIR evaluation script"
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to trained SwinIR .pth model"
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing degraded .npy images"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for restored outputs"
    )

    args = parser.parse_args()

    model_path = os.path.abspath(
        args.model_path
    )

    input_dir = os.path.abspath(
        args.input_dir
    )

    output_dir = os.path.abspath(
        args.output_dir
    )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Input directory not found:\n{input_dir}"
        )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("==========================================")
    print("       KLA SWINIR EVALUATION")
    print("==========================================")
    print("Device     :", device)

    if torch.cuda.is_available():
        print(
            "GPU        :",
            torch.cuda.get_device_name(0)
        )
    else:
        print("GPU        : CPU")

    print("Model      :", model_path)
    print("Input      :", input_dir)
    print("Output     :", output_dir)
    print("==========================================")
    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = build_model(
        model_path,
        device
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model loaded successfully."
    )

    print(
        "Parameters :",
        parameter_count
    )

    print()

    # --------------------------------------------------------
    # Find input files
    # --------------------------------------------------------

    files = sorted(
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith(".npy")
    )

    if not files:
        raise RuntimeError(
            "No .npy files found in input directory."
        )

    print(
        "Images found:",
        len(files)
    )

    print()

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    total_time = 0.0
    successful = 0

    with torch.no_grad():

        for index, filename in enumerate(
            files,
            start=1
        ):

            input_path = os.path.join(
                input_dir,
                filename
            )

            base_name = os.path.splitext(
                filename
            )[0]

            output_npy = os.path.join(
                output_dir,
                base_name + ".npy"
            )

            output_png = os.path.join(
                output_dir,
                base_name + ".png"
            )

            # ------------------------------------------------
            # Load input
            # ------------------------------------------------

            image = np.load(
                input_path
            ).astype(np.float32)

            image = np.squeeze(image)

            if image.shape != (128, 128):
                print(
                    f"Skipping {filename}: "
                    f"expected (128,128), "
                    f"got {image.shape}"
                )
                continue

            tensor = (
                torch.from_numpy(image)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            # ------------------------------------------------
            # Accurate timing
            # ------------------------------------------------

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            restored = model(tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - start_time
            )

            total_time += elapsed

            # ------------------------------------------------
            # Convert output
            # ------------------------------------------------

            restored = (
                restored
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )

            if restored.shape != (256, 256):
                print(
                    f"Skipping {filename}: "
                    f"unexpected output shape "
                    f"{restored.shape}"
                )
                continue

            restored = np.clip(
                restored,
                0.0,
                1.0
            )

            # ------------------------------------------------
            # Save float32 NPY
            # ------------------------------------------------

            np.save(
                output_npy,
                restored.astype(
                    np.float32
                )
            )

            # ------------------------------------------------
            # Save PNG
            # ------------------------------------------------

            save_png(
                restored,
                output_png
            )

            successful += 1

            print(
                f"[{index}/{len(files)}] "
                f"{filename} | "
                f"{elapsed * 1000:.3f} ms"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    if successful == 0:
        raise RuntimeError(
            "No images were successfully processed."
        )

    average_time = (
        total_time / successful
    )

    print()
    print("==========================================")
    print("        EVALUATION COMPLETE")
    print("==========================================")
    print(
        "Images processed :",
        successful
    )
    print(
        "Total inference  : "
        f"{total_time:.4f} s"
    )
    print(
        "Average/image    : "
        f"{average_time * 1000:.3f} ms"
    )
    print(
        "Outputs saved to :"
    )
    print(output_dir)
    print()
    print("Formats:")
    print("  .npy = float32 benchmark output")
    print("  .png = visual output")
    print("==========================================")


if __name__ == "__main__":
    main()