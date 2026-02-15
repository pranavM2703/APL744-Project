# ============================================================================
# Zenodo Dataset Downloader — Ti-6Al-4V Synthetic Microstructures
# ============================================================================
"""
Downloads 3D synthetic microstructure volumes from Zenodo (Stopka et al.)
and extracts 2D slices for the FGM inverse design pipeline.

The Ti-6Al-4V dataset consists of DREAM.3D-generated microstructural
instantiations containing alpha-phase grains and porosity, which are
used for microstructure-sensitive fatigue modelling.

Supported formats:
    - HDF5 / DREAM.3D (.dream3d, .h5)  → 3D voxel volumes
    - TIFF stacks (.tif, .tiff)         → 3D image stacks
    - Pre-sliced PNGs (.png)            → direct copy

Usage:
    python -m data.download_zenodo \
        --record-id 3610487 \
        --output-dir data/raw/images/ti64 \
        --num-slices 5000 \
        --slice-axis 2
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


# ── Zenodo API ──────────────────────────────────────────────────────────────

def download_zenodo_record(
    record_id: str,
    download_dir: str | Path,
    file_filter: Optional[str] = None,
) -> list[Path]:
    """Download all files from a Zenodo record.

    Parameters
    ----------
    record_id : str
        Zenodo record identifier (e.g., ``"3610487"``).
    download_dir : str or Path
        Local directory to save downloaded files.
    file_filter : str, optional
        Glob pattern to filter filenames (e.g., ``"*.h5"``).

    Returns
    -------
    list[Path]
        Paths to the downloaded files.
    """
    import urllib.request
    import json

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    # Query record metadata
    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"[INFO] Querying Zenodo record {record_id}...")

    with urllib.request.urlopen(api_url) as resp:
        metadata = json.loads(resp.read().decode())

    files = metadata.get("files", [])
    if not files:
        raise ValueError(f"No files found in Zenodo record {record_id}")

    downloaded: list[Path] = []
    for f in files:
        fname = f["key"]

        # Apply filter
        if file_filter and not Path(fname).match(file_filter):
            continue

        url = f["links"]["self"]
        dest = download_dir / fname

        if dest.exists():
            print(f"  [SKIP] {fname} (already exists)")
            downloaded.append(dest)
            continue

        print(f"  [DOWNLOAD] {fname} ({f['size'] / 1e6:.1f} MB)")
        urllib.request.urlretrieve(url, dest)
        downloaded.append(dest)

    return downloaded


# ── 3D Volume Slicing ───────────────────────────────────────────────────────

def extract_slices_from_hdf5(
    h5_path: str | Path,
    output_dir: str | Path,
    dataset_key: str = "DataContainers/ImageDataContainer/CellData/ImageData",
    num_slices: int = 5000,
    slice_axis: int = 2,
    seed: int = 42,
) -> int:
    """Extract 2D slices from a DREAM.3D / HDF5 volume.

    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file.
    output_dir : str or Path
        Directory to save PNG slices.
    dataset_key : str
        HDF5 dataset path inside the file. Common keys for DREAM.3D:
        - ``DataContainers/ImageDataContainer/CellData/ImageData``
        - ``DataContainers/ImageDataContainer/CellData/Phases``
    num_slices : int
        Maximum number of slices to extract.
    slice_axis : int
        Axis along which to slice (0=x, 1=y, 2=z).
    seed : int
        RNG seed for slice selection.

    Returns
    -------
    int
        Number of slices saved.
    """
    import h5py

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        # Auto-discover the dataset key if the default doesn't exist
        if dataset_key not in f:
            print(f"[WARN] Dataset key '{dataset_key}' not found. Searching...")
            dataset_key = _find_image_dataset(f)
            print(f"  → Found: '{dataset_key}'")

        volume = f[dataset_key][:]  # Load entire volume into memory

    # Normalise to [0, 1] float
    volume = volume.astype(np.float32)
    if volume.max() > 1.0:
        volume = volume / volume.max()

    # Determine available slices
    n_available = volume.shape[slice_axis]
    n_extract = min(num_slices, n_available)

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_available, size=n_extract, replace=n_extract > n_available)
    indices = np.sort(indices)

    saved = 0
    for i, idx in enumerate(tqdm(indices, desc="Extracting slices")):
        slc = np.take(volume, idx, axis=slice_axis)

        # Squeeze singleton dimensions if present
        slc = slc.squeeze()
        if slc.ndim != 2:
            continue

        # Binarize and save
        binary = (slc > 0.5).astype(np.uint8) * 255
        Image.fromarray(binary, mode="L").save(
            output_dir / f"ti64_{i:05d}.png"
        )
        saved += 1

    return saved


def extract_slices_from_tiff(
    tiff_path: str | Path,
    output_dir: str | Path,
    num_slices: int = 5000,
    seed: int = 42,
) -> int:
    """Extract 2D slices from a multi-page TIFF stack.

    Parameters
    ----------
    tiff_path : str or Path
        Path to the TIFF stack.
    output_dir : str or Path
        Directory to save PNG slices.
    num_slices : int
        Maximum number of slices to extract.
    seed : int
        RNG seed for slice selection.

    Returns
    -------
    int
        Number of slices saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(tiff_path)

    # Count frames
    n_frames = 0
    try:
        while True:
            n_frames += 1
            img.seek(n_frames)
    except EOFError:
        pass

    n_extract = min(num_slices, n_frames)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_frames, size=n_extract, replace=False))

    saved = 0
    for i, idx in enumerate(tqdm(indices, desc="Extracting TIFF slices")):
        img.seek(idx)
        arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
        binary = (arr > 0.5).astype(np.uint8) * 255
        Image.fromarray(binary, mode="L").save(
            output_dir / f"ti64_{i:05d}.png"
        )
        saved += 1

    return saved


def _find_image_dataset(h5_file) -> str:
    """Auto-discover the image/voxel dataset inside a DREAM.3D file."""
    candidates = []

    def _visitor(name, obj):
        import h5py
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
            candidates.append(name)

    h5_file.visititems(_visitor)

    if not candidates:
        raise KeyError(
            "No 3D datasets found in HDF5 file. "
            f"Available keys: {list(h5_file.keys())}"
        )

    # Prefer datasets with 'Image' or 'Phase' in the name
    for c in candidates:
        if "image" in c.lower() or "phase" in c.lower():
            return c
    return candidates[0]


# ── Pipeline: download + extract ────────────────────────────────────────────

def download_and_extract(
    record_id: str,
    output_dir: str | Path,
    num_slices: int = 5000,
    slice_axis: int = 2,
    seed: int = 42,
) -> None:
    """Download a Zenodo record and extract 2D microstructure slices.

    Parameters
    ----------
    record_id : str
        Zenodo record ID.
    output_dir : str or Path
        Final directory for the extracted PNG slices.
    num_slices : int
        Number of slices to extract.
    slice_axis : int
        Axis for slicing 3D volumes.
    seed : int
        RNG seed.
    """
    output_dir = Path(output_dir)
    raw_dir = output_dir.parent / "zenodo_raw"

    # Step 1: Download
    downloaded = download_zenodo_record(record_id, raw_dir)

    total_saved = 0
    for fpath in downloaded:
        suffix = fpath.suffix.lower()

        # Handle zip archives
        if suffix == ".zip":
            extract_dir = raw_dir / fpath.stem
            with zipfile.ZipFile(fpath, "r") as z:
                z.extractall(extract_dir)
            # Process extracted files recursively
            for inner in sorted(extract_dir.rglob("*")):
                total_saved += _process_single_file(
                    inner, output_dir, num_slices - total_saved, slice_axis, seed
                )
                if total_saved >= num_slices:
                    break

        else:
            total_saved += _process_single_file(
                fpath, output_dir, num_slices - total_saved, slice_axis, seed
            )

        if total_saved >= num_slices:
            break

    print(f"\n[DONE] Extracted {total_saved} slices → {output_dir}")


def _process_single_file(
    fpath: Path,
    output_dir: Path,
    remaining: int,
    slice_axis: int,
    seed: int,
) -> int:
    """Process a single downloaded file and return number of slices saved."""
    if remaining <= 0:
        return 0

    suffix = fpath.suffix.lower()

    if suffix in {".h5", ".hdf5", ".dream3d"}:
        return extract_slices_from_hdf5(
            fpath, output_dir, num_slices=remaining, slice_axis=slice_axis, seed=seed
        )
    elif suffix in {".tif", ".tiff"}:
        return extract_slices_from_tiff(
            fpath, output_dir, num_slices=remaining, seed=seed
        )
    elif suffix == ".png":
        output_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        dest = output_dir / fpath.name
        if not dest.exists():
            shutil.copy2(fpath, dest)
            return 1
    return 0


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Ti-6Al-4V microstructures from Zenodo and extract 2D slices."
    )
    parser.add_argument(
        "--record-id", type=str, required=True,
        help="Zenodo record ID (e.g. 3610487).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw/images/ti64",
        help="Directory to save extracted PNG slices.",
    )
    parser.add_argument(
        "--num-slices", type=int, default=5000,
        help="Number of 2D slices to extract.",
    )
    parser.add_argument(
        "--slice-axis", type=int, default=2, choices=[0, 1, 2],
        help="Axis along which to slice 3D volumes (0=x, 1=y, 2=z).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    download_and_extract(
        record_id=args.record_id,
        output_dir=args.output_dir,
        num_slices=args.num_slices,
        slice_axis=args.slice_axis,
        seed=args.seed,
    )
