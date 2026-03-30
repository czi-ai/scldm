# Training Benchmarks

Resource usage for VAE training across datasets, measured on 8x A100-80GB GPUs with DDP.
Data sourced from wandb projects [scg-vae/likelihood_vae_long](https://wandb.ai/scg-vae/likelihood_vae_long) and [scg-vae/dentate_gyrus_ae](https://wandb.ai/scg-vae/dentate_gyrus_ae).

## Gaussian Decoder

| Dataset | Time 8xA100 (h) | Time 1xA100 (h) | Peak Mem/GPU (GB) |
|---|---|---|---|
| dentate_gyrus | 0.30 ± 0.00 | 2.17 ± 0.00 | 32.5 |
| tabula_muris | 14.61 ± 0.19 | 106.0 | 34.10 |
| hlca | 17.01 ± 0.11 | 123.4 | 49.59 |
| replogle | 2.91 ± 0.02 | 21.1 | 5.92 |
| parse1m | 5.40 ± 0.01 | 39.2 | 5.92 |

## Negative Binomial (Shared Theta) Decoder

| Dataset | Time 8xA100 (h) | Time 1xA100 (h) | Peak Mem/GPU (GB) |
|---|---|---|---|
| dentate_gyrus | 0.30 ± 0.00 | 2.17 ± 0.00 | 32.5 |
| tabula_muris | 14.50 ± 0.28 | 105.2 | 33.18 |
| hlca | 16.97 ± 0.15 | 123.1 | 44.72 |
| replogle | 3.19 ± 0.04 | 23.1 | 5.83 |
| parse1m | 5.94 ± 0.04 | 43.1 | 5.83 |

## Notes

- 8xA100 values: mean ± std across 3 random seeds (12345, 24123, 37329).
- **dentate_gyrus 1xA100**: measured directly from single-GPU runs ([scg-vae/dentate_gyrus_ae](https://wandb.ai/scg-vae/dentate_gyrus_ae), n_embed=128, 3 seeds). Memory is also from 1-GPU runs.
- **Other datasets 1xA100**: estimated using a 7.25x scaling factor derived from comparing 8-GPU vs actual 1-GPU dentate_gyrus runs (vs naive 8x linear scaling).
- Peak memory for non-dentate datasets is per-GPU from 8-GPU DDP runs (actual 1-GPU memory would be higher due to full batch on a single device).
- GPU type: NVIDIA A100-80GB.
