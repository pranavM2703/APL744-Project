# ============================================================================
# Pipeline Validation Suite
# ============================================================================
"""
Validates the structural integrity and differentiability of all
pipeline components — data generators, model architectures, and
gradient flow — ensuring readiness for downstream SDS optimization.

Run:
    python -m pytest tests/test_pipeline_validation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Configuration ───────────────────────────────────────────────────────────

class TestConfigurationIntegrity:
    """Verify that the YAML configuration schema is valid and complete."""

    @pytest.fixture
    def config(self) -> dict:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_config_loads_without_error(self, config: dict) -> None:
        assert isinstance(config, dict)

    def test_required_sections_present(self, config: dict) -> None:
        required = {"data", "material", "vae", "surrogate", "training", "sds"}
        assert required.issubset(config.keys()), (
            f"Missing config sections: {required - config.keys()}"
        )

    def test_latent_channel_consistency(self, config: dict) -> None:
        assert config["vae"]["latent_channels"] == config["surrogate"]["input_channels"], (
            "VAE latent channels must match surrogate input channels."
        )

    def test_split_ratios_sum_to_one(self, config: dict) -> None:
        t = config["training"]
        total = t["train_split"] + t["val_split"] + t["test_split"]
        assert abs(total - 1.0) < 1e-6, f"Split ratios sum to {total}, expected 1.0"


# ── Data Generation ────────────────────────────────────────────────────────

class TestGaussianRandomFieldGenerator:
    """Validate the spectral GRF microstructure generator."""

    def test_output_shape(self) -> None:
        from data.generate_grf import generate_grf

        img = generate_grf(size=128)
        assert img.shape == (128, 128)

    def test_binary_output_domain(self) -> None:
        from data.generate_grf import generate_grf

        img = generate_grf(size=64)
        unique_vals = set(np.unique(img))
        assert unique_vals.issubset({0.0, 1.0}), (
            f"Expected binary output, got values: {unique_vals}"
        )

    def test_reproducibility_with_seed(self) -> None:
        from data.generate_grf import generate_grf

        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        img1 = generate_grf(size=64, rng=rng1)
        img2 = generate_grf(size=64, rng=rng2)
        np.testing.assert_array_equal(img1, img2)

    def test_volume_fraction_within_bounds(self) -> None:
        from data.generate_grf import generate_grf

        img = generate_grf(size=128, threshold=0.5)
        vf = np.mean(img)
        assert 0.0 < vf < 1.0, f"Volume fraction {vf} is degenerate."


# ── Physics Labeling ───────────────────────────────────────────────────────

class TestPhysicsHomogenization:
    """Validate the FE homogenization and volume fraction computations."""

    def test_volume_fraction_solid(self) -> None:
        from data.fe_homogenization import compute_volume_fraction

        solid = np.ones((64, 64), dtype=np.float32)
        assert abs(compute_volume_fraction(solid) - 1.0) < 1e-6

    def test_volume_fraction_void(self) -> None:
        from data.fe_homogenization import compute_volume_fraction

        void = np.zeros((64, 64), dtype=np.float32)
        assert abs(compute_volume_fraction(void)) < 1e-6

    def test_effective_modulus_monotonicity(self) -> None:
        from data.fe_homogenization import compute_effective_modulus

        sparse = np.zeros((64, 64), dtype=np.float32)
        sparse[:16, :] = 1.0  # ~25% solid

        dense = np.ones((64, 64), dtype=np.float32)
        dense[:16, :] = 0.0   # ~75% solid

        E_sparse = compute_effective_modulus(sparse)
        E_dense = compute_effective_modulus(dense)

        assert E_dense > E_sparse, (
            f"Monotonicity violated: E(75% solid)={E_dense} ≤ E(25% solid)={E_sparse}"
        )


# ── Surrogate Architecture ─────────────────────────────────────────────────

class TestLatentStiffnessRegressor:
    """Validate the surrogate network architecture and forward pass."""

    @pytest.fixture
    def model(self):
        from models.surrogate import LatentStiffnessRegressor
        return LatentStiffnessRegressor(input_channels=16, output_dim=1)

    def test_forward_output_shape(self, model) -> None:
        z = torch.randn(4, 16, 32, 32)
        out = model(z)
        assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"

    def test_single_sample_predict(self, model) -> None:
        z = torch.randn(1, 16, 32, 32)
        value = model.predict(z)
        assert isinstance(value, float)

    def test_parameter_count_is_reasonable(self, model) -> None:
        n_params = sum(p.numel() for p in model.parameters())
        # ResNet-18 ≈ 11M; our modified version should be in the same ballpark
        assert 1e6 < n_params < 50e6, f"Unexpected param count: {n_params}"


# ── Gradient Flow (SDS Feasibility) ────────────────────────────────────────

class TestGradientFlowVerification:
    """Verify differentiability of the surrogate w.r.t. latent inputs."""

    def test_nonzero_gradient_norm(self) -> None:
        from models.surrogate import LatentStiffnessRegressor

        model = LatentStiffnessRegressor(input_channels=16)
        model.eval()

        z = torch.randn(1, 16, 32, 32, requires_grad=True)
        pred = model(z)
        target = torch.tensor([[200.0]])
        loss = (pred - target).pow(2).mean()
        loss.backward()

        assert z.grad is not None, "Gradient is None — graph is broken."
        assert z.grad.norm().item() > 0, "Zero gradient — SDS will not converge."

    def test_gradient_shape_matches_input(self) -> None:
        from models.surrogate import LatentStiffnessRegressor

        model = LatentStiffnessRegressor(input_channels=16)
        z = torch.randn(2, 16, 32, 32, requires_grad=True)
        loss = model(z).sum()
        loss.backward()

        assert z.grad.shape == z.shape, (
            f"Gradient shape {z.grad.shape} ≠ input shape {z.shape}"
        )
