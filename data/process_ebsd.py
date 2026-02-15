# ============================================================================
# Process EBSD CTF Files → Binary Microstructure Images
# ============================================================================
"""
Parses EBSD Channel Text Files (.ctf) from the Zenodo Ti-6Al-4V dataset
(Record 15310081, Esmaeilzadeh et al. 2025) and produces 256×256
binary microstructure patches for the FGM inverse design pipeline.

The CTF format contains per-pixel crystallographic data:
    - Phase 0 = unindexed (void / porosity)
    - Phase 1 = BCC-Ti (beta phase)
    - Phase 2 = HCP-Ti (alpha phase)

The script constructs a phase map, binarizes it (solid=white, void=black),
and extracts overlapping 256×256 patches to generate a training dataset.

Usage:
    python -m data.process_ebsd \
        --input-dir data/raw/zenodo_raw/EBSD_extracted \
        --output-dir data/raw/images/ti64 \
        --patch-size 256 \
        --stride 64 \
        --augment
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_ctf(ctf_path: str | Path) -> tuple[np.ndarray, dict]:
    """Parse an EBSD Channel Text File (.ctf) into a phase map.

    Parameters
    ----------
    ctf_path : str or Path
        Path to the .ctf file.

    Returns
    -------
    phase_map : np.ndarray
        2D array of shape (YCells, XCells) with integer phase IDs.
    metadata : dict
        Header metadata including XCells, YCells, XStep, YStep, phases.
    """
    ctf_path = Path(ctf_path)
    metadata = {}
    header_lines = 0

    with open(ctf_path, "r") as f:
        for line in f:
            header_lines += 1
            line = line.strip()

            if line.startswith("XCells"):
                metadata["XCells"] = int(line.split()[-1])
            elif line.startswith("YCells"):
                metadata["YCells"] = int(line.split()[-1])
            elif line.startswith("XStep"):
                metadata["XStep"] = float(line.split()[-1])
            elif line.startswith("YStep"):
                metadata["YStep"] = float(line.split()[-1])
            elif line.startswith("Phase\tX\tY"):
                # This is the column header — data follows
                break

    xcells = metadata["XCells"]
    ycells = metadata["YCells"]

    # Read the data block — first column is the phase ID
    phases = []
    with open(ctf_path, "r") as f:
        for i, line in enumerate(f):
            if i < header_lines:
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 1:
                try:
                    phases.append(int(parts[0]))
                except ValueError:
                    phases.append(0)  # treat unparseable as void

    phase_array = np.array(phases, dtype=np.int32)

    # Reshape to 2D grid
    expected = xcells * ycells
    if len(phase_array) < expected:
        # Pad with zeros if data is shorter
        phase_array = np.pad(phase_array, (0, expected - len(phase_array)))
    elif len(phase_array) > expected:
        phase_array = phase_array[:expected]

    phase_map = phase_array.reshape(ycells, xcells)
    metadata["n_pixels"] = expected
    metadata["n_phases_found"] = len(set(phases))

    return phase_map, metadata


def binarize_phase_map(
    phase_map: np.ndarray,
    solid_phases: Optional[list[int]] = None,
) -> np.ndarray:
    """Convert a multi-phase map to a binary image.

    Parameters
    ----------
    phase_map : np.ndarray
        Integer phase map from ``parse_ctf()``.
    solid_phases : list[int], optional
        Phase IDs to treat as solid (white). Default: [1, 2] (both
        BCC-Ti and HCP-Ti are solid; phase 0 = void).

    Returns
    -------
    np.ndarray
        Binary image: 255 = solid, 0 = void.  dtype=uint8.
    """
    if solid_phases is None:
        solid_phases = [1, 2]

    binary = np.isin(phase_map, solid_phases).astype(np.uint8) * 255
    return binary


def extract_patches(
    image: np.ndarray,
    patch_size: int = 256,
    stride: int = 64,
    min_vf: float = 0.05,
    max_vf: float = 0.95,
) -> list[np.ndarray]:
    """Extract overlapping patches from a binary image.

    Patches with extreme volume fractions (nearly all-solid or all-void)
    are filtered out to ensure training diversity.

    Parameters
    ----------
    image : np.ndarray
        Binary image (uint8, values 0 or 255).
    patch_size : int
        Side length of square patches.
    stride : int
        Step size between patches.
    min_vf : float
        Minimum solid volume fraction to keep patch.
    max_vf : float
        Maximum solid volume fraction to keep patch.

    Returns
    -------
    list[np.ndarray]
        List of qualifying patches.
    """
    h, w = image.shape[:2]
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]
            vf = patch.mean() / 255.0

            if min_vf <= vf <= max_vf:
                patches.append(patch)

    return patches


def augment_patch(patch: np.ndarray) -> list[np.ndarray]:
    """Apply 8-fold dihedral augmentation (rotations + flips).

    Parameters
    ----------
    patch : np.ndarray
        Single 2D patch.

    Returns
    -------
    list[np.ndarray]
        8 augmented versions.
    """
    augmented = []
    for k in range(4):
        rotated = np.rot90(patch, k)
        augmented.append(rotated)
        augmented.append(np.fliplr(rotated))
    return augmented


def process_ebsd_to_patches(
    input_dir: str | Path,
    output_dir: str | Path,
    patch_size: int = 256,
    stride: int = 64,
    augment: bool = True,
    min_vf: float = 0.05,
    max_vf: float = 0.95,
) -> int:
    """Process all .ctf files in a directory into binary patches.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing .ctf files (searched recursively).
    output_dir : str or Path
        Directory to save PNG patches.
    patch_size : int
        Patch side length.
    stride : int
        Stride between patches.
    augment : bool
        Whether to apply 8-fold dihedral augmentation.
    min_vf : float
        Min volume fraction filter.
    max_vf : float
        Max volume fraction filter.

    Returns
    -------
    int
        Total number of patches saved.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctf_files = sorted(input_dir.rglob("*.ctf"))
    if not ctf_files:
        raise FileNotFoundError(f"No .ctf files found in {input_dir}")

    print(f"[INFO] Found {len(ctf_files)} CTF file(s)")

    total_saved = 0

    for ctf_path in ctf_files:
        print(f"\n[PROCESS] {ctf_path.name}")

        # Parse CTF → phase map
        phase_map, meta = parse_ctf(ctf_path)
        print(f"  Grid: {meta['XCells']}×{meta['YCells']} "
              f"({meta['n_phases_found']} phases)")

        # Binarize
        binary = binarize_phase_map(phase_map)
        overall_vf = binary.mean() / 255.0
        print(f"  Overall solid VF: {overall_vf:.3f}")

        # Extract patches
        patches = extract_patches(
            binary,
            patch_size=patch_size,
            stride=stride,
            min_vf=min_vf,
            max_vf=max_vf,
        )
        print(f"  Extracted {len(patches)} patches "
              f"(size={patch_size}, stride={stride})")

        # Augment and save
        for i, patch in enumerate(tqdm(patches, desc="  Saving patches")):
            if augment:
                aug_patches = augment_patch(patch)
            else:
                aug_patches = [patch]

            for j, ap in enumerate(aug_patches):
                fname = f"ti64_{ctf_path.stem}_{total_saved:05d}.png"
                Image.fromarray(ap, mode="L").save(output_dir / fname)
                total_saved += 1

    return total_saved


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert EBSD .ctf files to binary microstructure patches."
    )
    parser.add_argument(
        "--input-dir", type=str,
        default="data/raw/zenodo_raw/EBSD_extracted",
        help="Directory containing .ctf files.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="data/raw/images/ti64",
        help="Output directory for PNG patches.",
    )
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument(
        "--augment", action="store_true", default=True,
        help="Apply 8-fold dihedral augmentation.",
    )
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    parser.add_argument("--min-vf", type=float, default=0.05)
    parser.add_argument("--max-vf", type=float, default=0.95)
    args = parser.parse_args()

    n = process_ebsd_to_patches(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        augment=args.augment,
        min_vf=args.min_vf,
        max_vf=args.max_vf,
    )
    print(f"\n[DONE] Saved {n} patches → {args.output_dir}")
