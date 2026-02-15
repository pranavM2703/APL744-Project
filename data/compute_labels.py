# ============================================================================
# Physics Labeling — Compute Volume Fraction & Effective Modulus
# ============================================================================
"""
Iterates over a directory of binary microstructure images, computes
physics labels (Volume Fraction, Effective Elastic Modulus), and
writes the results to a structured CSV file.

Usage:
    python -m data.compute_labels \
        --image-dir data/raw/images \
        --output data/raw/labels.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .fe_homogenization import compute_effective_modulus, compute_volume_fraction


def compute_labels_for_directory(
    image_dir: str | Path,
    output_csv: str | Path,
    E_solid: float = 113.8,
    nu_solid: float = 0.342,
    E_void: float = 1e-3,
    nu_void: float = 0.0,
) -> pd.DataFrame:
    """Compute physics labels for all images in a directory.

    Parameters
    ----------
    image_dir : str or Path
        Directory containing PNG/TIFF microstructure images.
    output_csv : str or Path
        Path to write the resulting label CSV.
    E_solid, nu_solid, E_void, nu_void : float
        Material constants for homogenization.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: filename, volume_fraction, E_eff.
    """
    image_dir = Path(image_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in image_dir.rglob("*") if p.suffix.lower() in {".png", ".tiff", ".tif", ".jpg"}
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    records: list[dict] = []
    for img_path in tqdm(image_paths, desc="Computing physics labels"):
        img = np.array(Image.open(img_path).convert("L")).astype(np.float32) / 255.0
        binary = (img > 0.5).astype(np.float32)

        vf = compute_volume_fraction(binary)
        e_eff = compute_effective_modulus(binary, E_solid, nu_solid, E_void, nu_void)

        records.append({
            "filename": img_path.name,
            "relative_path": str(img_path.relative_to(image_dir)),
            "volume_fraction": round(vf, 6),
            "E_eff_GPa": round(e_eff, 4),
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Wrote {len(df)} labels to {output_csv}")
    return df


# ── entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute physics labels for microstructures.")
    parser.add_argument("--image-dir", type=str, default="data/raw/images")
    parser.add_argument("--output", type=str, default="data/raw/labels.csv")
    parser.add_argument("--E-solid", type=float, default=113.8)
    parser.add_argument("--nu-solid", type=float, default=0.342)
    args = parser.parse_args()

    compute_labels_for_directory(
        image_dir=args.image_dir,
        output_csv=args.output,
        E_solid=args.E_solid,
        nu_solid=args.nu_solid,
    )
