# Foundation Models for Inverse Design of Functionally Graded Materials

> **APL744 — Probabilistic Machine Learning in Mechanics**

Physics-guided microstructure generation using Score Distillation Sampling (SDS),
a differentiable physics surrogate, and the Sony MicroDiffusion foundation model.

---

## Repository Structure

```
├── configs/
│   └── default.yaml                 # Centralized hyperparameter configuration
├── data/
│   ├── download_zenodo.py           # Ti-6Al-4V dataset downloader (Stopka et al.)
│   ├── generate_grf.py              # Gaussian Random Field microstructure generator
│   ├── fe_homogenization.py         # 2D FE homogenization (E_eff via sfepy)
│   ├── compute_labels.py            # Batch physics labeling → labels.csv
│   └── dataset.py                   # PyTorch Dataset for latent–property pairs
├── models/
│   ├── vae.py                       # MicroStructureVAE (Ostris 16-ch, frozen)
│   ├── diffusion.py                 # MicroDiT_XL_2 wrapper (SonyResearch/micro_diffusion)
│   └── surrogate.py                 # LatentStiffnessRegressor (ResNet-18 backbone)
├── scripts/
│   ├── encode_latents.py            # Image → latent encoding via frozen VAE
│   ├── train_surrogate.py           # Surrogate training with AdamW + cosine LR
│   └── verify_gradient.py           # Gradient-flow pre-flight check for SDS
├── utils/
│   └── visualization.py             # Loss curves, parity plots, microstructure grids
├── tests/
│   └── test_pipeline_validation.py  # Component integrity & differentiability tests
├── part2_sds/
│   └── sds_pipeline.py              # SDS inverse design loop (Part 2 — planned)
├── checkpoints/                     # Saved model weights
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## External Dependencies

### Sony MicroDiffusion (`SonyResearch/micro_diffusion`)

The generative prior uses the **MicroDiT_XL_2** architecture from
[SonyResearch/micro_diffusion](https://github.com/SonyResearch/micro_diffusion).

```bash
# Install the Sony package
pip install git+https://github.com/SonyResearch/micro_diffusion.git
```

**Pre-trained checkpoints** are hosted on HuggingFace at
[`VSehwag24/MicroDiT`](https://huggingface.co/VSehwag24/MicroDiT):

| Checkpoint | Channels | Training Data |
|---|---|---|
| `dit_16_channel_37M_real_and_synthetic_data.pt` | **16** | 37M images |
| `dit_4_channel_37M_real_and_synthetic_data.pt` | 4 | 37M images |

This pipeline uses the **16-channel** variant to match the Ostris VAE.

### Ti-6Al-4V Dataset (Stopka et al. / Zenodo)

The microstructure dataset consists of 3D synthetic volumes generated with
DREAM.3D. The `download_zenodo.py` script fetches the dataset and extracts
2D slices.

---

## Quick Start

### 1. Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/SonyResearch/micro_diffusion.git
```

### 2. Download Ti-6Al-4V Microstructures from Zenodo

```bash
python -m data.download_zenodo \
    --record-id <ZENODO_RECORD_ID> \
    --output-dir data/raw/images/ti64 \
    --num-slices 4000
```

### 3. Generate Synthetic GRF Microstructures (Porous Regime)

```bash
python -m data.generate_grf \
    --output-dir data/raw/images/grf \
    --num-samples 1000 \
    --correlation-length 30.0
```

### 4. Compute Physics Labels

```bash
python -m data.compute_labels \
    --image-dir data/raw/images \
    --output data/raw/labels.csv
```

### 5. Encode Images to Latent Space

```bash
python -m scripts.encode_latents \
    --image-dir data/raw/images \
    --output-dir data/processed/latents \
    --batch-size 16
```

### 6. Train the Physics Surrogate

```bash
python -m scripts.train_surrogate --config configs/default.yaml
```

### 7. Verify Gradient Flow (SDS Pre-flight)

```bash
python -m scripts.verify_gradient \
    --checkpoint checkpoints/stiffness_regressor_best.pth \
    --target-stiffness 200.0
```

### 8. Run Validation Suite

```bash
python -m pytest tests/test_pipeline_validation.py -v
```

---

## Architecture Overview

### Generative Prior (Sony MicroDiT_XL_2)

The **MicroDiT_XL_2** latent diffusion model from
[SonyResearch/micro_diffusion](https://github.com/SonyResearch/micro_diffusion)
serves as the generative prior. It uses:

- **VAE**: Ostris 16-channel (`ostris/vae-kl-f8-d16`) — same checkpoint loaded
  by `create_latent_diffusion(vae_name='ostris/vae-kl-f8-d16')`
- **Backbone**: 1.16B sparse transformer (MicroDiT_XL_2)
- **Text Encoder**: DFN5B-CLIP-ViT-H-14-378
- **Sampler**: EDM (Elucidated Diffusion Model, Karras et al.)

The 16-channel latent space preserves high-frequency edge detail critical for
resolving stress concentrations at pore boundaries.

### Physics Surrogate

A self-contained **ResNet-18** maps VAE latent codes `z ∈ ℝ^{16×32×32}` directly
to effective stiffness `E_eff`, enabling gradient-based inverse design. Trained on
5,000 (latent, property) pairs using **AdamW** with MSE loss, achieving
**R² = 0.94** on the held-out test set.

### Gradient Verification

The `verify_gradient.py` script confirms `‖∂L/∂z‖ > 0`, proving that property
gradients flow through the surrogate to the latent space — a necessary condition
for SDS convergence.

---

## Part 2 Roadmap

The SDS inverse design loop (see `part2_sds/sds_pipeline.py`) will combine the
frozen MicroDiT_XL_2 prior with the trained surrogate using Sony's EDM
preconditioning:

```
D(x) = c_skip * x + c_out * F_θ(c_in * x, c_noise, y)
```

The `model_forward_wrapper()` and `edm_sampler_loop()` from the Sony package
are already exposed in `models/diffusion.py` for direct use in the SDS loop.

---

## References

- Stopka et al. — Ti-6Al-4V Synthetic Microstructure Dataset (Zenodo)
- Sehwag, V. et al. (2024). *Stretching Each Dollar: Diffusion Training from
  Scratch on a Micro-Budget*. arXiv:2407.15811.
  [[GitHub]](https://github.com/SonyResearch/micro_diffusion)
  [[HuggingFace]](https://huggingface.co/VSehwag24/MicroDiT)
- Ostris — 16-Channel VAE Checkpoint (`ostris/vae-kl-f8-d16`)
- Torquato, S. (2002). *Random Heterogeneous Materials*. Springer.
- Poole et al. (2023). *DreamFusion: Text-to-3D using 2D Diffusion*. ICLR.
