"""
Executes the full SDS inverse design pipeline:
  1. Loads the trained physics surrogate
  2. Trains (or loads) the denoising prior
  3. Runs the SDS loop for one or more target stiffness values
  4. Saves optimized latents and convergence plots

Usage:
    python -m scripts.run_sds \
        --checkpoint checkpoints/stiffness_regressor_best.pth \
        --target-stiffness 100 105 110 \
        --num-steps 200 \
        --lambda-physics 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.surrogate import LatentStiffnessRegressor
from part2_sds.sds_pipeline import (
    DenoiserWrapper,
    EDMNoiseSchedule,
    SimpleDenoisingPrior,
    multi_target_sds,
    sds_inverse_design_loop,
    train_simple_prior,
)


def plot_convergence(results: list[dict], save_path: str | Path) -> None:
    """Plot SDS convergence curves for all targets."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for res in results:
        h = res["history"]
        E_target = res["target"]
        steps = range(len(h["loss_physics"]))

        axes[0].plot(steps, h["loss_physics"], label=f"E*={E_target:.0f}")
        axes[1].plot(steps, h["pred_stiffness"], label=f"E*={E_target:.0f}")
        axes[2].plot(steps, h["sds_grad_norm"], label=f"E*={E_target:.0f}")

    # Physics loss
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("L_phys")
    axes[0].set_title("Physics Loss")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Predicted stiffness
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("E_pred (GPa)")
    axes[1].set_title("Predicted Stiffness")
    for res in results:
        axes[1].axhline(res["target"], color="gray", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # SDS gradient norm
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("‖∇SDS‖")
    axes[2].set_title("SDS Gradient Norm")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Convergence plot saved → {save_path}")


def plot_microstructures(results: list[dict], save_path: str | Path) -> None:
    """Visualize optimized latent codes (channel-0 slice)."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        z = res["z_optimized"].squeeze(0)  # (16, 32, 32)
        # Show first channel as grayscale
        ax.imshow(z[0].numpy(), cmap="viridis")
        pred_E = res["history"]["pred_stiffness"][-1]
        ax.set_title(f"E*={res['target']:.0f} | pred={pred_E:.1f} GPa", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Latent visualization saved → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run SDS inverse design.")
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/stiffness_regressor_best.pth",
        help="Path to trained surrogate checkpoint.",
    )
    parser.add_argument(
        "--target-stiffness", type=float, nargs="+",
        default=[95.0, 100.0, 105.0, 110.0, 115.0],
        help="Target stiffness values in GPa.",
    )
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--lambda-physics", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--latents-dir", type=str, default="data/processed/latents")
    parser.add_argument("--prior-epochs", type=int, default=30)
    parser.add_argument("--prior-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/sds")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load surrogate ──────────────────────────────────────────────
    print("\n[STEP 1] Loading trained surrogate...")
    surrogate = LatentStiffnessRegressor(input_channels=16, output_dim=1)
    ckpt_path = Path(args.checkpoint)

    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        surrogate.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: {ckpt_path}")
        print(f"  Epoch: {ckpt.get('epoch', '?')}, Val Loss: {ckpt.get('val_loss', '?')}")
    else:
        print(f"  [WARN] Checkpoint not found at {ckpt_path}")
        print(f"  Using randomly initialized surrogate (results will be approximate)")

    surrogate = surrogate.to(device).eval()

    # ── 2. Prepare denoiser ────────────────────────────────────────────
    print("\n[STEP 2] Preparing denoising prior...")
    schedule = EDMNoiseSchedule()

    if args.prior_checkpoint and Path(args.prior_checkpoint).exists():
        print(f"  Loading prior from {args.prior_checkpoint}")
        prior = SimpleDenoisingPrior()
        prior.load_state_dict(
            torch.load(args.prior_checkpoint, map_location=device, weights_only=True)
        )
        prior = prior.to(device).eval()
    else:
        print(f"  Training lightweight denoising prior on {args.latents_dir}...")
        prior = train_simple_prior(
            latents_dir=args.latents_dir,
            num_epochs=args.prior_epochs,
            device=device,
        )
        # Save for reuse
        prior_path = output_dir / "denoising_prior.pth"
        torch.save(prior.state_dict(), prior_path)
        print(f"  Saved prior checkpoint → {prior_path}")

    denoiser = DenoiserWrapper(model=prior, schedule=schedule, use_micro_dit=False)

    # ── 3. Run SDS ─────────────────────────────────────────────────────
    print(f"\n[STEP 3] Running SDS for targets: {args.target_stiffness}")
    results = multi_target_sds(
        surrogate=surrogate,
        denoiser=denoiser,
        targets=args.target_stiffness,
        lambda_physics=args.lambda_physics,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        device=device,
    )

    # ── 4. Save results ───────────────────────────────────────────────
    print(f"\n[STEP 4] Saving results...")

    # Save optimized latents
    for i, res in enumerate(results):
        torch.save(
            res["z_optimized"],
            output_dir / f"z_optimized_E{res['target']:.0f}.pt",
        )

    # Save convergence data
    summary = []
    for res in results:
        h = res["history"]
        summary.append({
            "target_GPa": res["target"],
            "final_pred_GPa": h["pred_stiffness"][-1],
            "final_loss_phys": h["loss_physics"][-1],
            "error_GPa": abs(h["pred_stiffness"][-1] - res["target"]),
        })

    print(f"\n{'='*60}")
    print(f"  SDS Results Summary")
    print(f"{'='*60}")
    print(f"  {'Target':>10s}  {'Predicted':>10s}  {'Error':>8s}")
    print(f"  {'(GPa)':>10s}  {'(GPa)':>10s}  {'(GPa)':>8s}")
    print(f"  {'-'*32}")
    for s in summary:
        print(f"  {s['target_GPa']:10.1f}  {s['final_pred_GPa']:10.2f}  {s['error_GPa']:8.2f}")
    print(f"{'='*60}")

    # Convergence plots
    plot_convergence(results, output_dir / "sds_convergence.png")
    plot_microstructures(results, output_dir / "latent_visualization.png")

    print(f"\n[DONE] All results saved to {output_dir}/")


if __name__ == "__main__":
    main()
