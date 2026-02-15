# ============================================================================
# PyTorch Dataset — Latent Representations with Physics Labels
# ============================================================================
"""
Provides a ``torch.utils.data.Dataset`` that pairs pre-computed VAE
latent tensors with physics labels (Volume Fraction, E_eff) from a CSV.

The dataset supports deterministic train / validation / test splits
seeded by a configurable random seed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LatentPropertyDataset(Dataset):
    """Map-style dataset of (latent, property) pairs.

    Parameters
    ----------
    latents_dir : str or Path
        Directory containing ``.pt`` files (each a latent tensor z).
    labels_csv : str or Path
        CSV with columns ``filename``, ``volume_fraction``, ``E_eff_GPa``.
    split : {"train", "val", "test"}
        Which split to return.
    train_ratio : float
        Fraction of data allocated to training.
    val_ratio : float
        Fraction of data allocated to validation.
    seed : int
        Random seed for deterministic splitting.
    target_key : str
        Column name of the regression target (default: ``E_eff_GPa``).
    """

    def __init__(
        self,
        latents_dir: str | Path,
        labels_csv: str | Path,
        split: Literal["train", "val", "test"] = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        target_key: str = "E_eff_GPa",
    ) -> None:
        super().__init__()
        self.latents_dir = Path(latents_dir)
        self.target_key = target_key

        # Load labels and sort for determinism
        df = pd.read_csv(labels_csv).sort_values("filename").reset_index(drop=True)

        # Match latent files to label rows
        latent_stems = {p.stem for p in self.latents_dir.glob("*.pt")}
        df["stem"] = df["filename"].apply(lambda f: Path(f).stem)
        df = df[df["stem"].isin(latent_stems)].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No matching latent/label pairs found. "
                f"Latents dir: {self.latents_dir}, CSV: {labels_csv}"
            )

        # Deterministic split
        n = len(df)
        indices = np.random.RandomState(seed).permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == "train":
            sel = indices[:n_train]
        elif split == "val":
            sel = indices[n_train : n_train + n_val]
        else:
            sel = indices[n_train + n_val :]

        self.df = df.iloc[sel].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        stem = row["stem"]

        # Load pre-encoded latent
        latent = torch.load(
            self.latents_dir / f"{stem}.pt", map_location="cpu", weights_only=True
        )

        # Scalar regression target
        target = torch.tensor([row[self.target_key]], dtype=torch.float32)

        return latent.squeeze(0) if latent.dim() == 4 else latent, target
