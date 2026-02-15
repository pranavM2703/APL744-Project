# ============================================================================
# Gaussian Random Field Generator
# ============================================================================
"""
Synthesizes binary microstructure images via thresholded Gaussian Random
Fields (GRFs).  The spectral method is used: a white-noise field is
convolved with a radially-symmetric power spectrum whose decay is
controlled by a correlation length parameter, producing spatially
correlated porosity patterns.

Reference:
    Torquato, S. (2002). Random Heterogeneous Materials. Springer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from PIL import Image
from tqdm import tqdm


# ── core generator ──────────────────────────────────────────────────────────

def generate_grf(
    size: int = 256,
    correlation_length: float = 30.0,
    threshold: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a single binary microstructure via thresholded GRF.

    Parameters
    ----------
    size : int
        Edge length of the square image (pixels).
    correlation_length : float
        Spatial correlation length controlling feature scale.
    threshold : float
        Quantile for binarization (0 = fully solid, 1 = fully void).
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Binary image of shape ``(size, size)`` with values in {0, 1}.
    """
    if rng is None:
        rng = np.random.default_rng()

    # White noise in spatial domain
    white_noise = rng.standard_normal((size, size))

    # Build radial power spectrum in frequency domain
    freq_x = fftfreq(size)
    freq_y = fftfreq(size)
    kx, ky = np.meshgrid(freq_x, freq_y, indexing="ij")
    k_mag = np.sqrt(kx ** 2 + ky ** 2)
    k_mag[0, 0] = 1.0  # avoid division by zero at DC

    # Gaussian power spectral density
    psd = np.exp(-2.0 * (np.pi * correlation_length * k_mag) ** 2)

    # Convolve in Fourier space → correlated field
    correlated = np.real(ifft2(fft2(white_noise) * np.sqrt(psd)))

    # Normalize to [0, 1] and binarize
    correlated -= correlated.min()
    correlated /= correlated.max() + 1e-12
    binary = (correlated > threshold).astype(np.float32)

    return binary


# ── batch generation CLI ────────────────────────────────────────────────────

def generate_batch(
    output_dir: str | Path,
    num_samples: int = 1000,
    size: int = 256,
    correlation_length: float = 30.0,
    threshold: float = 0.5,
    seed: int = 42,
) -> None:
    """Generate and save a batch of GRF microstructures as PNG images.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save images.
    num_samples : int
        Number of samples to generate.
    size : int
        Edge length of each image (pixels).
    correlation_length : float
        GRF correlation length.
    threshold : float
        Binarization threshold.
    seed : int
        RNG seed for reproducibility.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i in tqdm(range(num_samples), desc="Generating GRF microstructures"):
        img = generate_grf(size, correlation_length, threshold, rng)
        img_uint8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_uint8, mode="L").save(output_dir / f"grf_{i:05d}.png")


# ── entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic GRF microstructures."
    )
    parser.add_argument("--output-dir", type=str, default="data/raw/images/grf")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--correlation-length", type=float, default=30.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_batch(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        size=args.size,
        correlation_length=args.correlation_length,
        threshold=args.threshold,
        seed=args.seed,
    )
