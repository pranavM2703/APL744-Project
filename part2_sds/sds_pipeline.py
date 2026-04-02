# ============================================================================
# Part 2 — Score Distillation Sampling (SDS) Pipeline
# ============================================================================
"""
Implements the SDS-based inverse design loop for generating Functionally
Graded Materials (FGMs) with prescribed effective stiffness.

Mathematical Foundation (EDM Formulation):
    Given a pre-trained denoiser D_θ and a physics surrogate f_θ, we
    optimize a latent code z ∈ ℝ^{16×32×32} via:

        z^(t+1) = z^(t) − α · (∇_z L_SDS + λ · ∇_z L_phys)

    where:
        L_SDS  = w(σ) · ‖D_θ(z_σ, σ, y) − z‖²     (score distillation)
        L_phys = ‖f_θ(z) − E*‖²                      (property matching)

    The SDS gradient pushes z toward high-probability regions of the
    learned microstructure manifold, while L_phys steers it toward the
    target stiffness.

Integration with Sony MicroDiffusion:
    When ``micro_diffusion`` is installed, the pipeline uses
    MicroDiT_XL_2 as the denoiser. Otherwise, a lightweight
    SimpleDenoisingPrior (4-layer U-Net) serves as a fallback, ensuring
    the pipeline runs end-to-end on any machine.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
# EDM Noise Schedule (Karras et al. 2022)
# ════════════════════════════════════════════════════════════════════════════

class EDMNoiseSchedule:
    """Implements the EDM noise schedule from Karras et al. (2022).

    The schedule defines a sequence of noise levels σ_i via:
        σ_i = (σ_max^{1/ρ} + i/(N-1) · (σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ

    Parameters
    ----------
    sigma_min : float
        Minimum noise level (default: 0.002).
    sigma_max : float
        Maximum noise level (default: 80).
    rho : float
        Schedule curvature parameter (default: 7).
    sigma_data : float
        Data standard deviation for preconditioning (default: 0.9).
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.9,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data

    def get_sigmas(self, num_steps: int) -> torch.Tensor:
        """Generate the full noise schedule.

        Returns
        -------
        torch.Tensor
            Decreasing sequence [σ_0, σ_1, ..., σ_N, 0] of shape (N+1,).
        """
        ramp = torch.linspace(0, 1, num_steps)
        min_inv = self.sigma_min ** (1 / self.rho)
        max_inv = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv + ramp * (min_inv - max_inv)) ** self.rho
        return torch.cat([sigmas, sigmas.new_zeros(1)])  # append σ=0

    def sample_sigma(self, batch_size: int, device: str = "cpu") -> torch.Tensor:
        """Sample random noise levels from the EDM training distribution.

        ln(σ) ~ N(P_mean, P_std²) with P_mean=-0.6, P_std=1.2
        """
        ln_sigma = torch.randn(batch_size, device=device) * 1.2 - 0.6
        sigma = ln_sigma.exp().clamp(self.sigma_min, self.sigma_max)
        return sigma

    def edm_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """EDM loss weight: w(σ) = (σ² + σ_data²) / (σ · σ_data)²"""
        sd = self.sigma_data
        return (sigma**2 + sd**2) / (sigma * sd) ** 2

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        """Skip connection scaling: c_skip(σ) = σ_data² / (σ² + σ_data²)"""
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        """Output scaling: c_out(σ) = σ · σ_data / √(σ² + σ_data²)"""
        return sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        """Input scaling: c_in(σ) = 1 / √(σ² + σ_data²)"""
        return 1.0 / (sigma**2 + self.sigma_data**2).sqrt()

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        """Noise conditioning: c_noise(σ) = ln(σ) / 4"""
        return sigma.log() / 4.0


# ════════════════════════════════════════════════════════════════════════════
# Lightweight Denoising Prior (Fallback when MicroDiT unavailable)
# ════════════════════════════════════════════════════════════════════════════

class _ResBlock(nn.Module):
    """Residual block with timestep conditioning."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.sigma_proj = nn.Linear(1, channels)

    def forward(self, x: torch.Tensor, sigma_emb: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        # Add sigma conditioning as channel-wise bias
        h = h + self.sigma_proj(sigma_emb).unsqueeze(-1).unsqueeze(-1)
        return x + h


class SimpleDenoisingPrior(nn.Module):
    """Lightweight U-Net style denoiser for latent space.

    This serves as a fallback when Sony's MicroDiT is not available.
    It learns a simple prior over the latent space from the encoded
    training data and can be used for SDS optimization.

    Architecture:
        4-layer residual network operating on 16×32×32 latents
        with sigma-conditioned residual blocks.

    Parameters
    ----------
    channels : int
        Number of latent channels (default: 16).
    hidden : int
        Hidden channel width (default: 64).
    num_blocks : int
        Number of residual blocks (default: 4).
    """

    def __init__(
        self,
        channels: int = 16,
        hidden: int = 64,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Conv2d(channels, hidden, 1)
        self.blocks = nn.ModuleList([_ResBlock(hidden) for _ in range(num_blocks)])
        self.proj_out = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Predict noise residual.

        Parameters
        ----------
        x : torch.Tensor
            Noisy latent (B, C, H, W).
        sigma : torch.Tensor
            Noise level (B,) or (B, 1).

        Returns
        -------
        torch.Tensor
            Predicted clean signal component (B, C, H, W).
        """
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        sigma_emb = sigma.view(-1, 1).float()

        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h, sigma_emb)
        return self.proj_out(h)


def train_simple_prior(
    latents_dir: str | Path,
    num_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
) -> SimpleDenoisingPrior:
    """Train the lightweight denoising prior on encoded latents.

    Uses a denoising score matching objective:
        L = w(σ) · ‖D_θ(z + σε, σ) − z‖²

    Parameters
    ----------
    latents_dir : str or Path
        Directory containing .pt latent files.
    num_epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Training batch size.
    device : str
        Compute device.

    Returns
    -------
    SimpleDenoisingPrior
        Trained denoiser.
    """
    latents_dir = Path(latents_dir)
    pt_files = sorted(latents_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {latents_dir}")

    # Load all latents into memory
    all_latents = []
    for f in pt_files:
        z = torch.load(f, map_location="cpu", weights_only=True)
        if z.dim() == 4:
            z = z.squeeze(0)
        all_latents.append(z)
    all_latents = torch.stack(all_latents)  # (N, 16, 32, 32)
    print(f"[INFO] Loaded {len(all_latents)} latents, shape: {all_latents.shape}")

    schedule = EDMNoiseSchedule()
    model = SimpleDenoisingPrior(channels=all_latents.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    n = len(all_latents)
    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(n)
        total_loss = 0.0
        num_batches = 0

        for i in range(0, n, batch_size):
            batch = all_latents[perm[i:i + batch_size]].to(device)
            B = batch.shape[0]

            sigma = schedule.sample_sigma(B, device=device)
            noise = torch.randn_like(batch)
            noisy = batch + sigma.view(B, 1, 1, 1) * noise

            # EDM preconditioning
            c_skip = schedule.c_skip(sigma).view(B, 1, 1, 1)
            c_out = schedule.c_out(sigma).view(B, 1, 1, 1)
            c_in = schedule.c_in(sigma).view(B, 1, 1, 1)

            # Network predicts F_θ, denoised estimate = c_skip·x + c_out·F_θ(c_in·x)
            F_out = model(c_in * noisy, sigma)
            denoised = c_skip * noisy + c_out * F_out

            weight = schedule.edm_weight(sigma).view(B, 1, 1, 1)
            loss = (weight * (denoised - batch) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [Prior] Epoch {epoch:3d}/{num_epochs} | Loss: {avg_loss:.6f}")

    model.eval()
    return model


# ════════════════════════════════════════════════════════════════════════════
# Denoiser Wrapper (unifies MicroDiT and SimpleDenoisingPrior)
# ════════════════════════════════════════════════════════════════════════════

class DenoiserWrapper:
    """Unified denoiser interface for the SDS loop.

    Wraps either Sony MicroDiT or SimpleDenoisingPrior behind the
    same API: denoise(z_noisy, sigma) → z_denoised.
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: EDMNoiseSchedule,
        use_micro_dit: bool = False,
    ) -> None:
        self.model = model
        self.schedule = schedule
        self.use_micro_dit = use_micro_dit

    @torch.no_grad()
    def denoise(
        self,
        z_noisy: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute denoised estimate D_θ(z_σ, σ).

        D_θ(x, σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x, c_noise(σ))

        Parameters
        ----------
        z_noisy : torch.Tensor
            Noisy latent (B, 16, 32, 32).
        sigma : torch.Tensor
            Noise level (B,).

        Returns
        -------
        torch.Tensor
            Denoised prediction (B, 16, 32, 32).
        """
        B = z_noisy.shape[0]
        s = self.schedule

        if self.use_micro_dit:
            # Use Sony's model_forward_wrapper which handles preconditioning
            result = self.model.model_forward_wrapper(z_noisy, sigma, y=None)
            return result['sample']
        else:
            # Manual EDM preconditioning with SimpleDenoisingPrior
            c_skip = s.c_skip(sigma).view(B, 1, 1, 1)
            c_out = s.c_out(sigma).view(B, 1, 1, 1)
            c_in = s.c_in(sigma).view(B, 1, 1, 1)

            F_out = self.model(c_in * z_noisy, sigma)
            return c_skip * z_noisy + c_out * F_out


# ════════════════════════════════════════════════════════════════════════════
# Target Stiffness Functions
# ════════════════════════════════════════════════════════════════════════════

def linear_gradient_target(y: float, E_min: float = 93.0, E_max: float = 117.0) -> float:
    """Linear stiffness gradient along the y-axis.

    E*(y) = E_min + (E_max − E_min) · y,  y ∈ [0, 1]
    """
    return E_min + (E_max - E_min) * y


def radial_gradient_target(
    x: float, y: float,
    E_center: float = 117.0, E_edge: float = 93.0,
) -> float:
    """Radial stiffness gradient from center to edge.

    E*(r) = E_center + (E_edge − E_center) · r,  r = √((x−0.5)² + (y−0.5)²) · √2
    """
    r = math.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) * math.sqrt(2)
    r = min(r, 1.0)
    return E_center + (E_edge - E_center) * r


def constant_target(E: float = 105.0) -> Callable:
    """Factory for constant stiffness target."""
    return lambda *_: E


# ════════════════════════════════════════════════════════════════════════════
# SDS Gradient Computation
# ════════════════════════════════════════════════════════════════════════════

def compute_sds_gradient(
    z: torch.Tensor,
    denoiser: DenoiserWrapper,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Compute the Score Distillation Sampling gradient.

    ∇_z L_SDS = w(σ) · (D_θ(z + σε, σ) − z)

    Parameters
    ----------
    z : torch.Tensor
        Current latent (B, 16, 32, 32). Requires grad.
    denoiser : DenoiserWrapper
        Wrapped denoiser model.
    sigma : torch.Tensor
        Noise level (B,).

    Returns
    -------
    torch.Tensor
        SDS gradient, same shape as z.
    """
    B = z.shape[0]
    device = z.device

    # Add noise
    epsilon = torch.randn_like(z)
    z_noisy = z.detach() + sigma.view(B, 1, 1, 1) * epsilon

    # Get denoised estimate (no grad through denoiser)
    z_denoised = denoiser.denoise(z_noisy, sigma)

    # SDS gradient: w(σ) · (z_denoised − z)
    weight = denoiser.schedule.edm_weight(sigma).view(B, 1, 1, 1)
    grad_sds = weight * (z_denoised - z.detach())

    return grad_sds


def compute_physics_gradient(
    z: torch.Tensor,
    surrogate: nn.Module,
    target_stiffness: float,
) -> tuple[torch.Tensor, float, float]:
    """Compute the physics loss gradient.

    L_phys = (f_θ(z) − E*)²
    ∇_z L_phys = 2(f_θ(z) − E*) · ∂f_θ/∂z

    Parameters
    ----------
    z : torch.Tensor
        Current latent (1, 16, 32, 32). Must have requires_grad=True.
    surrogate : nn.Module
        Trained LatentStiffnessRegressor.
    target_stiffness : float
        Target E* in GPa.

    Returns
    -------
    grad : torch.Tensor
        Physics gradient, same shape as z.
    loss_val : float
        Scalar physics loss value.
    pred_val : float
        Current predicted stiffness.
    """
    z_input = z.detach().requires_grad_(True)
    pred = surrogate(z_input)  # (1, 1)
    loss = (pred.squeeze() - target_stiffness) ** 2
    loss.backward()
    return z_input.grad.detach(), loss.item(), pred.item()


# ════════════════════════════════════════════════════════════════════════════
# Main SDS Inverse Design Loop
# ════════════════════════════════════════════════════════════════════════════

def sds_inverse_design_loop(
    surrogate: nn.Module,
    denoiser: DenoiserWrapper,
    target_stiffness: float = 105.0,
    lambda_physics: float = 10.0,
    lambda_sds: float = 0.001,
    num_steps: int = 200,
    learning_rate: float = 0.005,
    sigma_range: tuple[float, float] = (0.05, 1.0),
    grad_clip: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Run the SDS inverse design loop.

    Optimizes z to minimize:
        L = L_SDS(z) + λ · L_phys(z)

    using gradient descent with the SDS trick for the diffusion prior
    and standard backprop through the surrogate for the physics loss.

    Parameters
    ----------
    surrogate : nn.Module
        Trained LatentStiffnessRegressor.
    denoiser : DenoiserWrapper
        Wrapped denoiser (MicroDiT or SimpleDenoisingPrior).
    target_stiffness : float
        Target E_eff in GPa.
    lambda_physics : float
        Relative weight of physics loss.
    num_steps : int
        Number of optimization steps.
    learning_rate : float
        Step size for latent updates.
    sigma_range : tuple[float, float]
        Range of noise levels to sample from during SDS.
    seed : int
        Random seed.
    device : str
        Compute device.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Contains:
        - 'z_optimized': Final latent (1, 16, 32, 32)
        - 'loss_history': List of total loss per step
        - 'physics_loss_history': Physics loss per step
        - 'sds_loss_history': SDS gradient norm per step
        - 'pred_history': Predicted stiffness per step
        - 'target': Target stiffness
    """
    torch.manual_seed(seed)

    # Initialize latent from standard normal
    z = torch.randn(1, 16, 32, 32, device=device) * 0.5
    z.requires_grad_(False)  # We do manual gradient updates

    surrogate = surrogate.to(device).eval()

    history = {
        "loss_total": [],
        "loss_physics": [],
        "sds_grad_norm": [],
        "pred_stiffness": [],
        "sigma_used": [],
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"SDS Inverse Design Loop")
        print(f"{'='*60}")
        print(f"  Target E*    : {target_stiffness:.2f} GPa")
        print(f"  λ_physics    : {lambda_physics}")
        print(f"  λ_sds        : {lambda_sds}")
        print(f"  Steps        : {num_steps}")
        print(f"  LR           : {learning_rate}")
        print(f"  σ range      : [{sigma_range[0]}, {sigma_range[1]}]")
        print(f"  Grad clip    : {grad_clip}")
        print(f"{'='*60}\n")

    for step in range(num_steps):
        # ── Anneal sigma: high noise early → low noise late ────────────
        progress = step / max(num_steps - 1, 1)
        sigma_val = sigma_range[1] * (1 - progress) + sigma_range[0] * progress
        sigma = torch.tensor([sigma_val], device=device)

        # ── Compute SDS gradient ──────────────────────────────────────
        grad_sds = compute_sds_gradient(z, denoiser, sigma)
        sds_norm = grad_sds.norm().item()

        # ── Compute physics gradient ──────────────────────────────────
        grad_phys, loss_phys, pred_E = compute_physics_gradient(
            z, surrogate, target_stiffness
        )

        # ── Combined gradient with clipping ───────────────────────────
        total_grad = lambda_sds * grad_sds + lambda_physics * grad_phys
        # Clip gradient to prevent divergence
        grad_norm = total_grad.norm()
        if grad_norm > grad_clip:
            total_grad = total_grad * (grad_clip / grad_norm)
        z = z - learning_rate * total_grad

        # ── Record ────────────────────────────────────────────────────
        total_loss = sds_norm + lambda_physics * loss_phys
        history["loss_total"].append(total_loss)
        history["loss_physics"].append(loss_phys)
        history["sds_grad_norm"].append(sds_norm)
        history["pred_stiffness"].append(pred_E)
        history["sigma_used"].append(sigma_val)

        if verbose and (step % 20 == 0 or step == num_steps - 1):
            print(
                f"  Step {step:4d}/{num_steps} | "
                f"σ={sigma_val:.4f} | "
                f"E_pred={pred_E:.2f} GPa | "
                f"L_phys={loss_phys:.4f} | "
                f"‖∇SDS‖={sds_norm:.4f}"
            )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Final prediction: {history['pred_stiffness'][-1]:.2f} GPa")
        print(f"  Target:           {target_stiffness:.2f} GPa")
        print(f"  Error:            {abs(history['pred_stiffness'][-1] - target_stiffness):.2f} GPa")
        print(f"{'='*60}")

    return {
        "z_optimized": z.detach().cpu(),
        "history": history,
        "target": target_stiffness,
    }


# ════════════════════════════════════════════════════════════════════════════
# Multi-Target SDS (generate microstructures for a range of stiffnesses)
# ════════════════════════════════════════════════════════════════════════════

def multi_target_sds(
    surrogate: nn.Module,
    denoiser: DenoiserWrapper,
    targets: list[float],
    **kwargs,
) -> list[dict]:
    """Run SDS for multiple target stiffness values.

    Parameters
    ----------
    surrogate : nn.Module
        Trained surrogate.
    denoiser : DenoiserWrapper
        Wrapped denoiser.
    targets : list[float]
        List of target stiffness values in GPa.
    **kwargs
        Additional arguments passed to sds_inverse_design_loop.

    Returns
    -------
    list[dict]
        Results for each target.
    """
    results = []
    for i, E_target in enumerate(targets):
        print(f"\n[TARGET {i+1}/{len(targets)}] E* = {E_target:.2f} GPa")
        result = sds_inverse_design_loop(
            surrogate=surrogate,
            denoiser=denoiser,
            target_stiffness=E_target,
            seed=42 + i,
            **kwargs,
        )
        results.append(result)
    return results
