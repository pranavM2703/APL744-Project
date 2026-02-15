# ============================================================================
# Surrogate Training — Latent → Stiffness Regression
# ============================================================================
"""
End-to-end training loop for the ``LatentStiffnessRegressor``.

Reads pre-computed VAE latents and physics labels, trains the surrogate
with AdamW + cosine annealing, logs metrics to TensorBoard, and saves
the best checkpoint.

Usage:
    python -m scripts.train_surrogate --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import LatentPropertyDataset
from models.surrogate import LatentStiffnessRegressor


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def train(config_path: str) -> None:
    """Run the full training pipeline.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    """
    # ── load configuration ──────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    surr_cfg = cfg["surrogate"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(train_cfg["seed"])

    # ── data loaders ────────────────────────────────────────────────────
    common_ds_kwargs = dict(
        latents_dir=data_cfg["latents_dir"],
        labels_csv=data_cfg["labels_path"],
        train_ratio=train_cfg["train_split"],
        val_ratio=train_cfg["val_split"],
        seed=train_cfg["seed"],
    )

    train_ds = LatentPropertyDataset(split="train", **common_ds_kwargs)
    val_ds = LatentPropertyDataset(split="val", **common_ds_kwargs)
    test_ds = LatentPropertyDataset(split="test", **common_ds_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    print(f"[INFO] Dataset splits — Train: {len(train_ds)} | "
          f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── model, optimizer, scheduler ─────────────────────────────────────
    model = LatentStiffnessRegressor(
        input_channels=surr_cfg["input_channels"],
        output_dim=surr_cfg["output_dim"],
        pretrained_backbone=surr_cfg["pretrained_backbone"],
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["epochs"] - train_cfg["warmup_epochs"],
    )

    # Optional: TensorBoard logging
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        log_dir = Path(train_cfg["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
    except ImportError:
        print("[WARN] TensorBoard not available — skipping metric logging.")

    # ── training loop ───────────────────────────────────────────────────
    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, train_cfg["epochs"] + 1):
        # ── train ───────────────────────────────────────────────────────
        model.train()
        train_losses: list[float] = []

        for latents, targets in train_loader:
            latents = latents.to(device)
            targets = targets.to(device)

            preds = model(latents)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Learning rate schedule (skip warmup epochs)
        if epoch > train_cfg["warmup_epochs"]:
            scheduler.step()

        avg_train_loss = np.mean(train_losses)

        # ── validate ────────────────────────────────────────────────────
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for latents, targets in val_loader:
                latents = latents.to(device)
                targets = targets.to(device)
                preds = model(latents)
                val_losses.append(criterion(preds, targets).item())

        avg_val_loss = np.mean(val_losses)

        # ── logging ─────────────────────────────────────────────────────
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:3d}/{train_cfg['epochs']}] | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"LR: {lr:.2e}"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("LR", lr, epoch)

        # ── checkpoint ──────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = ckpt_dir / "stiffness_regressor_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                },
                ckpt_path,
            )
            print(f"  ↳ Saved best checkpoint → {ckpt_path}")

    # ── final evaluation on test set ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Final Evaluation on Held-Out Test Set")
    print("=" * 60)

    # Reload best weights
    best_ckpt = torch.load(ckpt_dir / "stiffness_regressor_best.pth", weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    all_preds: list[float] = []
    all_targets: list[float] = []

    with torch.no_grad():
        for latents, targets in test_loader:
            latents = latents.to(device)
            preds = model(latents)
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_targets.extend(targets.numpy().flatten().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    test_mse = np.mean((y_true - y_pred) ** 2)
    test_r2 = compute_r2(y_true, y_pred)

    print(f"  Test MSE : {test_mse:.6f}")
    print(f"  Test R²  : {test_r2:.4f}")

    if writer is not None:
        writer.add_scalar("Test/MSE", test_mse, train_cfg["epochs"])
        writer.add_scalar("Test/R2", test_r2, train_cfg["epochs"])
        writer.close()

    # Save final checkpoint
    torch.save(
        {
            "epoch": train_cfg["epochs"],
            "model_state_dict": model.state_dict(),
            "test_mse": test_mse,
            "test_r2": test_r2,
        },
        ckpt_dir / "stiffness_regressor_final.pth",
    )
    print(f"\n[DONE] Final checkpoint saved to {ckpt_dir}/stiffness_regressor_final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the latent stiffness surrogate.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
