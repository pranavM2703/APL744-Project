# ============================================================================
# Models — Package Initializer
# ============================================================================
"""
Neural network architectures for the FGM inverse design pipeline:

- ``MicroStructureVAE``: Frozen Ostris 16-ch encoder/decoder
  (same checkpoint as Sony MicroDiffusion).
- ``LatentStiffnessRegressor``: Differentiable physics surrogate.
- ``MicroDiffusionWrapper``: Sony MicroDiT_XL_2 latent diffusion model.
"""

from .vae import MicroStructureVAE
from .surrogate import LatentStiffnessRegressor

# MicroDiffusionWrapper requires the micro_diffusion package;
# defer import to avoid hard dependency at package level.
__all__ = [
    "MicroStructureVAE",
    "LatentStiffnessRegressor",
    "MicroDiffusionWrapper",
]


def __getattr__(name: str):
    """Lazy import for MicroDiffusionWrapper to avoid import errors
    when ``micro_diffusion`` is not installed."""
    if name == "MicroDiffusionWrapper":
        from .diffusion import MicroDiffusionWrapper
        return MicroDiffusionWrapper
    raise AttributeError(f"module 'models' has no attribute {name!r}")
