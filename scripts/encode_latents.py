# ============================================================================
# Latent Encoding — Batch Encode Images via Frozen VAE
# ============================================================================
"""
Reads microstructure images from disk, passes them through the frozen
Ostris VAE encoder, and saves each latent representation as an
individual ``.pt`` file for downstream surrogate training.

Usage:
    python -m scripts.encode_latents \
        --image-dir data/raw/images \
        --output-dir data/processed/latents \
        --batch-size 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vae import MicroStructureVAE


def encode_directory(
    image_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 16,
    image_size: int = 256,
    device: str = "cpu",
) -> None:
    """Encode all images in a directory to VAE latents.

    Parameters
    ----------
    image_dir : str or Path
        Directory containing microstructure images.
    output_dir : str or Path
        Directory to save ``.pt`` latent files.
    batch_size : int
        Number of images to encode simultaneously.
    image_size : int
        Resize target for each image edge.
    device : str
        Compute device.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in {".png", ".tiff", ".tif", ".jpg"}
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Load the frozen VAE
    vae = MicroStructureVAE(device=device)

    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding latents"):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        stems = []

        for p in batch_paths:
            img = Image.open(p).convert("L")
            images.append(transform(img))
            stems.append(p.stem)

        batch_tensor = torch.stack(images).to(device)  # (B, 1, H, W)
        latents = vae.encode(batch_tensor)              # (B, 16, H/8, W/8)

        # Save each latent individually
        for j, stem in enumerate(stems):
            torch.save(latents[j].cpu(), output_dir / f"{stem}.pt")

    print(f"[INFO] Encoded {len(image_paths)} images → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-encode images to VAE latents.")
    parser.add_argument("--image-dir", type=str, default="data/raw/images")
    parser.add_argument("--output-dir", type=str, default="data/processed/latents")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    encode_directory(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device,
    )
