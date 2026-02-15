# ============================================================================
# MicroStructure VAE — Frozen Encoder / Decoder Wrapper
# ============================================================================
"""
Wraps the Ostris 16-channel Variational Autoencoder — the same VAE
used by the Sony MicroDiffusion pipeline (``SonyResearch/micro_diffusion``)
— behind a clean interface for encoding microstructure images to
latent representations and decoding them back.

Architecture details:
    - Input:  (B, 1, 256, 256) grayscale microstructure images
    - Latent: (B, 16, 32, 32) — 8× spatial compression, 16 channels
    - The 16-channel latent preserves high-frequency edge information
      critical for resolving stress concentrations at pore boundaries.

VAE checkpoint:
    HuggingFace: ``ostris/vae-kl-f8-d16``
    This is the same checkpoint loaded by Sony's
    ``create_latent_diffusion(vae_name='ostris/vae-kl-f8-d16')``
    in the ``micro_diffusion`` package.

All encoder/decoder weights are **frozen** — no gradients flow through
this module.  The latent codes it produces serve as inputs to the
trainable physics surrogate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class MicroStructureVAE(nn.Module):
    """Frozen VAE wrapper for microstructure encoding and decoding.

    Uses the same Ostris 16-channel checkpoint as the Sony MicroDiffusion
    pipeline.  The VAE is loaded via ``diffusers.AutoencoderKL`` exactly
    as ``micro_diffusion.models.model.create_latent_diffusion()`` does::

        AutoencoderKL.from_pretrained(
            'ostris/vae-kl-f8-d16',
            subfolder=None,   # no 'vae' subfolder for Ostris
        )

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier for the pretrained VAE checkpoint.
        Default matches Sony's ``create_latent_diffusion`` path.
    scaling_factor : float
        Latent-space scaling factor applied after encoding.  The value
        is read from ``vae.config.scaling_factor`` at runtime; the
        default here matches the Ostris-16ch config.
    device : str
        Target device (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model_id: str = "ostris/vae-kl-f8-d16",
        scaling_factor: float | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device

        try:
            from diffusers import AutoencoderKL

            # Load exactly as Sony's micro_diffusion does:
            #   AutoencoderKL.from_pretrained(vae_name, subfolder=None, ...)
            self.vae = AutoencoderKL.from_pretrained(
                model_id,
                subfolder=None,  # Ostris checkpoint is at the repo root
            ).to(device)

            # Use the VAE's own scaling factor (matches micro_diffusion)
            self.scaling_factor = (
                scaling_factor
                if scaling_factor is not None
                else self.vae.config.scaling_factor
            )

        except (ImportError, OSError) as exc:
            # Graceful degradation: create a lightweight stub when
            # diffusers or the checkpoint are unavailable (e.g., in CI).
            import warnings

            warnings.warn(
                f"Could not load pretrained VAE ({exc}). "
                "Falling back to a deterministic stub encoder/decoder. "
                "This is suitable for testing but NOT for production.",
                stacklevel=2,
            )
            self.vae = None
            self.scaling_factor = scaling_factor if scaling_factor else 0.13025
            self._build_stub()

        # Freeze all parameters — this model is inference-only
        self.requires_grad_(False)
        self.eval()

    # ── public API ──────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images to latent representations.

        Follows the same encoding path as ``LatentDiffusion.forward()``
        in ``micro_diffusion.models.model``::

            latents = vae.encode(images)['latent_dist'].sample()
            latents *= scaling_factor

        Parameters
        ----------
        image : torch.Tensor
            Batch of images, shape ``(B, C, H, W)``.  If single-channel,
            the channel dimension is replicated to 3 for the VAE.

        Returns
        -------
        torch.Tensor
            Latent code of shape ``(B, 16, H//8, W//8)``.
        """
        image = image.to(self.device)
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        if self.vae is not None:
            posterior = self.vae.encode(image)["latent_dist"]
            z = posterior.sample() * self.scaling_factor
        else:
            z = self._stub_encode(image)

        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent representation back to image space.

        Follows the same decoding path as ``LatentDiffusion.generate()``::

            image = vae.decode(latents / scaling_factor).sample

        Parameters
        ----------
        z : torch.Tensor
            Latent code of shape ``(B, 16, H//8, W//8)``.

        Returns
        -------
        torch.Tensor
            Reconstructed image of shape ``(B, 3, H, W)``.
        """
        z = z.to(self.device)

        if self.vae is not None:
            decoded = self.vae.decode(z / self.scaling_factor).sample
        else:
            decoded = self._stub_decode(z)

        return decoded

    # ── stub for environments without diffusers ─────────────────────────

    def _build_stub(self) -> None:
        """Create a lightweight convolutional stub for testing."""
        self._stub_enc = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=8, padding=0),
        )
        self._stub_dec = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=8, stride=8, padding=0),
            nn.Sigmoid(),
        )

    def _stub_encode(self, image: torch.Tensor) -> torch.Tensor:
        return self._stub_enc(image) * self.scaling_factor

    def _stub_decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._stub_dec(z / self.scaling_factor)
