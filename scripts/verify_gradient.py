# ============================================================================
# Gradient-Flow Verification — Pre-flight Check for SDS (Part 2)
# ============================================================================
"""
Verifies that gradients flow correctly from the surrogate loss back
to the input latent vector.  This is a necessary condition for
Score Distillation Sampling: if ∂L/∂z = 0, the diffusion-guided
optimization loop cannot modify the generated microstructure.

Usage:
    python -m scripts.verify_gradient \
        --checkpoint checkpoints/stiffness_regressor_best.pth \
        --target-stiffness 200.0

Expected output:
    ✓ Gradient flow verified — ‖∂L/∂z‖ > 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.surrogate import LatentStiffnessRegressor


def verify_gradient_flow(
    checkpoint_path: str | Path | None = None,
    target_stiffness: float = 200.0,
    latent_shape: tuple[int, ...] = (1, 16, 32, 32),
    device: str = "cpu",
) -> dict[str, float]:
    """Run the gradient-flow diagnostic.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to a trained surrogate checkpoint. If ``None``, uses a
        randomly-initialised model (still valid for verifying the
        computational graph).
    target_stiffness : float
        Desired stiffness target E* [GPa] for the SDS loss.
    latent_shape : tuple
        Shape of the synthetic latent input.
    device : str
        Compute device.

    Returns
    -------
    dict
        ``predicted``, ``target``, ``loss``, ``grad_norm`` — all floats.
    """
    # ── load model ──────────────────────────────────────────────────────
    model = LatentStiffnessRegressor(input_channels=latent_shape[1]).to(device)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print(f"[INFO] Loaded checkpoint from {checkpoint_path}")

    model.eval()  # BatchNorm in eval mode for stable gradients

    # ── forward pass with gradient tracking on the latent ───────────────
    z = torch.randn(latent_shape, device=device, requires_grad=True)

    prediction = model(z)
    target = torch.tensor([[target_stiffness]], device=device)

    loss = (prediction - target).pow(2).mean()  # MSE w.r.t. target
    loss.backward()

    grad_norm = z.grad.norm().item()

    # ── report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Gradient-Flow Verification Report")
    print("=" * 60)
    print(f"  Latent shape       : {list(latent_shape)}")
    print(f"  Predicted E_eff    : {prediction.item():.4f} GPa")
    print(f"  Target E*          : {target_stiffness:.1f} GPa")
    print(f"  MSE loss           : {loss.item():.4f}")
    print(f"  ‖∂L/∂z‖           : {grad_norm:.6f}")
    print()

    if grad_norm > 0:
        print("  ✓  Gradient flow verified — SDS is feasible.")
    else:
        print("  ✗  WARNING: Zero gradient detected.  Check model architecture.")

    return {
        "predicted": prediction.item(),
        "target": target_stiffness,
        "loss": loss.item(),
        "grad_norm": grad_norm,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify gradient flow through the physics surrogate."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained surrogate checkpoint (.pth).",
    )
    parser.add_argument(
        "--target-stiffness", type=float, default=200.0,
        help="Target stiffness E* for the SDS loss [GPa].",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    verify_gradient_flow(
        checkpoint_path=args.checkpoint,
        target_stiffness=args.target_stiffness,
        device=args.device,
    )
