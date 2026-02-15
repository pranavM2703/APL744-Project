# ============================================================================
# MicroDiffusion Wrapper — Sony MicroDiT_XL_2 Latent Diffusion Model
# ============================================================================
"""
Integrates the Sony MicroDiffusion generative prior into the FGM
inverse design pipeline.

This module wraps the ``SonyResearch/micro_diffusion`` package to
provide:
  - MicroDiT_XL_2 transformer backbone (1.16B sparse parameters)
  - Ostris 16-channel VAE (shared with ``models.vae``)
  - DFN5B-CLIP text encoder (for conditional generation)
  - EDM sampler loop (Elucidated Diffusion Model, Karras et al.)

Pre-trained checkpoint:
    Repository: https://huggingface.co/VSehwag24/MicroDiT
    File:       dit_16_channel_37M_real_and_synthetic_data.pt
    License:    Apache 2.0

Reference:
    Sehwag, V. et al. (2024). Stretching Each Dollar: Diffusion Training
    from Scratch on a Micro-Budget. arXiv:2407.15811.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ── Checkpoint download utility ─────────────────────────────────────────────

def download_dit_checkpoint(
    repo_id: str = "VSehwag24/MicroDiT",
    filename: str = "ckpts/dit_16_channel_37M_real_and_synthetic_data.pt",
    cache_dir: str | Path | None = None,
) -> Path:
    """Download the pre-trained MicroDiT checkpoint from HuggingFace.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository identifier.
    filename : str
        Path to the checkpoint file within the repository.
    cache_dir : str or Path, optional
        Local cache directory. Defaults to HuggingFace cache.

    Returns
    -------
    Path
        Local path to the downloaded checkpoint.
    """
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(local_path)


# ── MicroDiffusion Wrapper ──────────────────────────────────────────────────

class MicroDiffusionWrapper(nn.Module):
    """High-level wrapper around Sony's ``LatentDiffusion`` model.

    Provides a simplified interface for:
        1. Loading the MicroDiT_XL_2 diffusion model with pre-trained weights
        2. Generating microstructure images from text prompts
        3. Exposing the EDM sampler loop for Part 2 SDS integration

    Parameters
    ----------
    dit_arch : str
        DiT architecture name (default: ``MicroDiT_XL_2``).
    vae_name : str
        HuggingFace VAE checkpoint ID (default: Ostris 16-ch).
    text_encoder_name : str
        Text encoder identifier for CLIP embeddings.
    in_channels : int
        Number of latent channels (must be 16 for the 16-ch pipeline).
    latent_res : int
        Spatial resolution of the latent (256px image → 32 latent).
    checkpoint_path : str or Path, optional
        Path to a pre-trained DiT checkpoint. If ``None``, downloads
        the default 16-channel checkpoint from HuggingFace.
    device : str
        Compute device.
    dtype : str
        Precision for model operations.
    """

    def __init__(
        self,
        dit_arch: str = "MicroDiT_XL_2",
        vae_name: str = "ostris/vae-kl-f8-d16",
        text_encoder_name: str = "openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378",
        in_channels: int = 16,
        latent_res: int = 32,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.latent_res = latent_res

        try:
            from micro_diffusion.models.model import create_latent_diffusion

            self.model = create_latent_diffusion(
                vae_name=vae_name,
                text_encoder_name=text_encoder_name,
                dit_arch=dit_arch,
                latent_res=latent_res,
                in_channels=in_channels,
                pos_interp_scale=1.0,
                dtype=dtype,
                precomputed_latents=False,
            ).to(device)

            # Load pre-trained DiT weights
            if checkpoint_path is not None:
                self._load_dit_checkpoint(checkpoint_path)
            else:
                try:
                    ckpt_path = download_dit_checkpoint()
                    self._load_dit_checkpoint(ckpt_path)
                except Exception as exc:
                    import warnings
                    warnings.warn(
                        f"Could not download DiT checkpoint ({exc}). "
                        "Model initialized with random weights.",
                        stacklevel=2,
                    )

            self._micro_diffusion_available = True

        except ImportError:
            import warnings
            warnings.warn(
                "micro_diffusion package not installed. "
                "Install via: pip install git+https://github.com/"
                "SonyResearch/micro_diffusion.git\n"
                "Falling back to a stub — generation will not work.",
                stacklevel=2,
            )
            self._micro_diffusion_available = False
            self.model = None

        # Freeze all parameters — this is inference-only
        self.requires_grad_(False)
        self.eval()

    def _load_dit_checkpoint(self, path: str | Path) -> None:
        """Load pre-trained weights into the DiT backbone."""
        state = torch.load(str(path), map_location=self.device, weights_only=True)
        # Sony checkpoints store the DiT weights directly
        if "state_dict" in state:
            state = state["state_dict"]
        self.model.dit.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded MicroDiT checkpoint from {path}")

    # ── Public API ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: list[str],
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = 42,
    ) -> torch.Tensor:
        """Generate microstructure images from text prompts.

        Parameters
        ----------
        prompt : list[str]
            List of text prompts describing desired microstructures.
        num_inference_steps : int
            Number of EDM denoising steps.
        guidance_scale : float
            Classifier-free guidance scale.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
            Generated images of shape ``(B, 3, H, W)`` in [0, 1].
        """
        if not self._micro_diffusion_available:
            raise RuntimeError(
                "micro_diffusion is not installed. Cannot generate images."
            )

        return self.model.generate(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

    @property
    def vae(self):
        """Access the underlying VAE (shared with MicroStructureVAE)."""
        if self.model is not None:
            return self.model.vae
        return None

    @property
    def dit(self):
        """Access the DiT backbone for SDS gradient injection."""
        if self.model is not None:
            return self.model.dit
        return None

    @property
    def edm_config(self):
        """Access the EDM noise schedule configuration."""
        if self.model is not None:
            return self.model.edm_config
        return None

    def edm_sampler_loop(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        steps: int = 18,
        cfg: float = 1.0,
    ) -> torch.Tensor:
        """Run the EDM sampler loop (for Part 2 SDS integration).

        This is a thin wrapper around ``LatentDiffusion.edm_sampler_loop``
        that will be called inside the SDS optimization to obtain the
        denoised estimate ẑ₀ at each timestep.

        Parameters
        ----------
        x : torch.Tensor
            Initial noisy latent of shape ``(B, C, H, W)``.
        y : torch.Tensor
            Text/conditioning embeddings.
        steps : int
            Number of EDM denoising steps.
        cfg : float
            Classifier-free guidance scale.

        Returns
        -------
        torch.Tensor
            Denoised latent of shape ``(B, C, H, W)``.
        """
        if not self._micro_diffusion_available:
            raise RuntimeError("micro_diffusion is not installed.")

        return self.model.edm_sampler_loop(x, y, steps=steps, cfg=cfg)

    def model_forward_wrapper(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        y: torch.Tensor,
        mask_ratio: float = 0.0,
    ) -> dict:
        """Single denoising step (for SDS gradient computation).

        Wraps ``LatentDiffusion.model_forward_wrapper`` which applies
        the EDM preconditioning (c_skip, c_out, c_in, c_noise) and
        returns the denoised prediction.

        Parameters
        ----------
        x : torch.Tensor
            Noisy latent at noise level sigma.
        sigma : torch.Tensor
            Current noise level.
        y : torch.Tensor
            Conditioning embeddings.
        mask_ratio : float
            Patch masking ratio (0 for generation).

        Returns
        -------
        dict
            Contains ``'sample'`` key with denoised prediction D(x).
        """
        if not self._micro_diffusion_available:
            raise RuntimeError("micro_diffusion is not installed.")

        return self.model.model_forward_wrapper(
            x, sigma, y,
            model_forward_fxn=self.model.dit.forward,
            mask_ratio=mask_ratio,
        )
