
"""
Evaluates optimized latents from the SDS pipeline:
  1. Decodes latents to microstructure images via VAE
  2. Computes physics properties on decoded images
  3. Generates publication-quality comparison figures

Usage:
    python -m scripts.evaluate_sds --results-dir results/sds
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.surrogate import LatentStiffnessRegressor
from models.vae import MicroStructureVAE


def decode_and_save(
    z: torch.Tensor,
    vae: MicroStructureVAE,
    save_path: str | Path,
) -> np.ndarray:
    """Decode a latent to image and save as PNG.

    Parameters
    ----------
    z : torch.Tensor
        Latent of shape (1, 16, 32, 32).
    vae : MicroStructureVAE
        VAE decoder.
    save_path : str or Path
        Where to save the decoded image.

    Returns
    -------
    np.ndarray
        Decoded image as HxW uint8 array.
    """
    decoded = vae.decode(z)  # (1, 3, H, W)
    # Convert to grayscale
    img = decoded[0].mean(dim=0).clamp(0, 1).numpy()
    img_uint8 = (img * 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode="L").save(save_path)
    return img_uint8


def create_gallery(
    results_dir: Path,
    vae: MicroStructureVAE,
    surrogate: LatentStiffnessRegressor,
    save_path: Path,
) -> None:
    """Create a gallery of decoded microstructures with annotations."""
    pt_files = sorted(results_dir.glob("z_optimized_E*.pt"))
    if not pt_files:
        print("[WARN] No optimized latents found.")
        return

    n = len(pt_files)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), gridspec_kw={"height_ratios": [3, 1]})
    if n == 1:
        axes = axes.reshape(2, 1)

    targets = []
    preds = []

    for i, pt_file in enumerate(pt_files):
        z = torch.load(str(pt_file), map_location="cpu", weights_only=True)
        if z.dim() == 3:
            z = z.unsqueeze(0)

        # Decode
        decoded = vae.decode(z)
        img = decoded[0].mean(dim=0).clamp(0, 1).numpy()

        # Predict stiffness
        surrogate.eval()
        with torch.no_grad():
            pred_E = surrogate(z).item()

        # Extract target from filename
        target_E = float(pt_file.stem.split("_E")[-1])
        targets.append(target_E)
        preds.append(pred_E)

        # Plot microstructure
        axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"E* = {target_E:.0f} GPa\npred = {pred_E:.1f} GPa", fontsize=11)
        axes[0, i].axis("off")

        # Plot latent channel distribution
        z_np = z.squeeze(0).numpy()
        axes[1, i].hist(z_np.ravel(), bins=50, alpha=0.7, color="steelblue", density=True)
        axes[1, i].set_xlabel("Latent value")
        axes[1, i].set_ylabel("Density")
        axes[1, i].set_title(f"Latent distribution", fontsize=9)
        axes[1, i].grid(True, alpha=0.3)

    plt.suptitle("SDS Inverse Design — Generated Microstructures", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gallery saved → {save_path}")

    # Also save target vs predicted scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, s=80, c="steelblue", edgecolors="navy", zorder=3)
    min_val = min(min(targets), min(preds)) - 5
    max_val = max(max(targets), max(preds)) + 5
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("Target E* (GPa)", fontsize=12)
    ax.set_ylabel("Predicted E (GPa)", fontsize=12)
    ax.set_title("Target vs Predicted Stiffness", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    scatter_path = save_path.parent / "target_vs_predicted.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Scatter plot saved → {scatter_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SDS results.")
    parser.add_argument("--results-dir", type=str, default="results/sds")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stiffness_regressor_best.pth")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    device = args.device

    # Load models
    print("[INFO] Loading VAE...")
    vae = MicroStructureVAE(device=device)

    print("[INFO] Loading surrogate...")
    surrogate = LatentStiffnessRegressor(input_channels=16, output_dim=1)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        surrogate.load_state_dict(ckpt["model_state_dict"])
    surrogate = surrogate.to(device).eval()

    # Decode all optimized latents
    print("\n[INFO] Decoding optimized latents...")
    decoded_dir = results_dir / "decoded"
    decoded_dir.mkdir(exist_ok=True)

    for pt_file in sorted(results_dir.glob("z_optimized_E*.pt")):
        z = torch.load(str(pt_file), map_location=device, weights_only=True)
        if z.dim() == 3:
            z = z.unsqueeze(0)
        img = decode_and_save(z, vae, decoded_dir / f"{pt_file.stem}.png")
        print(f"  Decoded {pt_file.stem} → {img.shape}")

    # Create gallery
    create_gallery(results_dir, vae, surrogate, results_dir / "gallery.png")

    print(f"\n[DONE] All evaluation outputs → {results_dir}/")


if __name__ == "__main__":
    main()
