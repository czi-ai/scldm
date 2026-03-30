# Training Benchmarks

Resource usage for VAE training across datasets, measured on 8x A100-80GB GPUs with DDP.
Data sourced from wandb project [scg-vae/likelihood_vae_long](https://wandb.ai/scg-vae/likelihood_vae_long) (3 seeds per configuration: 12345, 24123, 37329).

## Gaussian Decoder

| Dataset | Time 8xA100 (h) | Time 1xA100 (h) | Peak Mem/GPU (GB) |
|---|---|---|---|
| czb_cd4_naive_holdout | 5.40 ± 0.01 | 43.2 ± 0.1 | 5.92 |
| dentate_gyrus | 0.30 ± 0.00 | 2.4 ± 0.0 | 28.97 |
| hlca | 17.01 ± 0.11 | 136.1 ± 0.9 | 49.59 |
| replogle | 2.91 ± 0.02 | 23.3 ± 0.2 | 5.92 |
| tabula_muris | 14.61 ± 0.19 | 116.9 ± 1.5 | 34.10 |

## Negative Binomial (Shared Theta) Decoder

| Dataset | Time 8xA100 (h) | Time 1xA100 (h) | Peak Mem/GPU (GB) |
|---|---|---|---|
| czb_cd4_naive_holdout | 5.94 ± 0.04 | 47.5 ± 0.3 | 5.83 |
| dentate_gyrus | 0.30 ± 0.00 | 2.4 ± 0.0 | 28.21 |
| hlca | 16.97 ± 0.15 | 135.7 ± 1.2 | 44.72 |
| replogle | 3.19 ± 0.04 | 25.5 ± 0.3 | 5.83 |
| tabula_muris | 14.50 ± 0.28 | 116.0 ± 2.2 | 33.18 |

## Notes

- All values reported as mean ± std across 3 random seeds.
- 1xA100 time is estimated by linear scaling (wall time x 8) from the 8-GPU DDP runs.
- Peak memory is per-GPU (all 8 GPUs show identical allocation in DDP).
- GPU type: NVIDIA A100-80GB.
