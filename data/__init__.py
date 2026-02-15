# ============================================================================
# Data Pipeline — Package Initializer
# ============================================================================
"""
Data generation, physics labeling, and dataset utilities for the FGM
inverse design pipeline.
"""

from .generate_grf import generate_grf
from .dataset import LatentPropertyDataset
from .download_zenodo import download_and_extract

__all__ = ["generate_grf", "LatentPropertyDataset", "download_and_extract"]
