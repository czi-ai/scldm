# Benchmark: Cell Reconstruction on dentate_gyrus

## Paper Results (from Table 1)

| Dataset | Model | Params | RE ↓ | PCC ↑ | MSE ↓ |
|---------|-------|--------|------|-------|-------|
| Dentate Gyrus | scVI | — | 5193.2 ± 0.1 | 0.058 ± 0.000 | 0.378 ± 0.000 |
| Dentate Gyrus | CFGen | — | 5468.8 ± N/A | 0.076 ± N/A | 0.253 ± N/A |
| Dentate Gyrus | scLDM (NB) | — | **4571.6** ± 26.5 | **0.273** ± 0.005 | **0.206** ± 0.002 |
| Tabula Muris | scVI | — | 5588.2 ± 1.0 | 0.221 ± 0.000 | 0.132 ± 0.000 |
| Tabula Muris | CFGen | — | 5547.6 ± N/A | 0.136 ± N/A | 0.127 ± N/A |
| Tabula Muris | scLDM (NB) | — | **4993.6** ± 25.1 | **0.376** ± 0.006 | **0.106** ± 0.001 |
| HLCA | scVI | — | 5659.2 ± 0.3 | 0.125 ± 0.000 | 0.238 ± 0.000 |
| HLCA | CFGen | — | 5428.7 ± N/A | 0.146 ± N/A | 0.117 ± N/A |
| HLCA | scLDM (NB) | — | **4898.9** ± 12.4 | **0.310** ± 0.003 | **0.095** ± 0.001 |

## 3-Fold CV Reproduction (dentate_gyrus, this repo)

Training: 500 epochs, batch_size=128, AdamWLegacy lr=1e-3, 3-fold KFold on training set (14,570 cells).

### ScviVAE — MLP-based (90.7M params)

Architecture: `EncoderScvi` (2-layer MLP) → `GaussianLinearLayer` → `DecoderScvi` (2-layer MLP) → `NegativeBinomialLinearLayer` (shared theta). n_hidden=2048, n_latent=2048.

| Fold | RE (test_llh) | PCC | MSE |
|------|-------------|-----|-----|
| 0 | 5539.7 | 0.0590 | 0.3677 |
| 1 | 5533.9 | 0.0578 | 0.3657 |
| 2 | 5577.2 | 0.0595 | 0.3669 |
| **Mean ± SE** | **5550.3 ± 11.1** | **0.059 ± 0.000** | **0.367 ± 0.000** |

Checkpoints: `scvi_dentate_gyrus_cv/fold_{0,1,2}/`
Script: `experiments/scripts/train_cv_scvi.py`

### TransformerVAE — Transformer-based (3.4M params)

Architecture: Set-Transformer `Encoder` → `Decoder` → `NegativeBinomialTransformerLayer` (shared theta). n_inducing_points=128, n_embed=128, n_embed_latent=16, n_layer=2, n_head=4. Latent dim = 128 × 16 = 2048. No KL term.

| Fold | RE (test_llh) | PCC | MSE |
|------|-------------|-----|-----|
| 0 | 4380.8 | 0.2717 | 0.2792 |
| 1 | 4455.2 | 0.2652 | 0.2806 |
| 2 | 4503.0 | 0.2550 | 0.2849 |
| **Mean ± SE** | **4446.3 ± 29.0** | **0.264 ± 0.004** | **0.282 ± 0.001** |

Checkpoints: `vae_dentate_gyrus_cv/fold_{0,1,2}/`
Script: `experiments/scripts/train_cv.py`

## Summary

| Model | Params | RE ↓ | PCC ↑ | MSE ↓ |
|-------|--------|------|-------|-------|
| ScviVAE (MLP) | 90.7M | 5550.3 ± 11.1 | 0.059 ± 0.000 | 0.367 ± 0.000 |
| **TransformerVAE** | **3.4M** | **4446.3 ± 29.0** | **0.264 ± 0.004** | **0.282 ± 0.001** |

The TransformerVAE achieves significantly better reconstruction with **26× fewer parameters** than the ScviVAE. The ScviVAE shows signs of overfitting (train loss ~4430 vs val RE ~5550) despite the large hidden/latent dimensions.
