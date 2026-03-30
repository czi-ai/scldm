# VAE 3-Fold CV on dentate_gyrus (NB likelihood)

## Architecture
- **Model**: TransformerVAE with NegativeBinomial decoder (shared_theta)
- **Latent dim**: 128 × 16 = 2048 (n_inducing_points × n_embed_latent)
- **Encoder/Decoder**: n_layer=2, n_embed=128, n_head=4, n_head_cross=1
- **Parameters**: 3.4M
- **Gene sampling**: expressed, genes_seq_len=6147

## Training
- **Dataset**: dentategyrus_train.h5ad (14,570 cells × 17,002 genes)
- **CV**: 3-fold KFold (sklearn), shuffle=True, seed=12345
- **Epochs**: 500 per fold
- **Batch size**: 128
- **Optimizer**: AdamWLegacy, lr=1e-3, betas=(0.9, 0.95)
- **Scheduler**: wsd_schedule (warmup-sqrt-decay), 10% warmup, 10% decay
- **Precision**: float32
- **Gradient clipping**: 10.0 (norm)

## Results

| Fold | Val Loss | Val MSE | Val PCC | Val Zeros Acc |
|------|----------|---------|---------|---------------|
| 0    | 4380.80  | 0.2792  | 0.2717  | 0.9041        |
| 1    | 4455.15  | 0.2806  | 0.2652  | 0.9024        |
| 2    | 4502.99  | 0.2849  | 0.2550  | 0.9012        |
| **Mean** | **4446.31** | **0.2816** | **0.2640** | **0.9026** |

Val metrics are computed on the held-out CV fold. PCC is on NB-sampled reconstructions.

## Checkpoints
- `fold_0/last.ckpt`, `fold_0/epoch=249.ckpt` (best val_loss)
- `fold_1/last.ckpt`, `fold_1/epoch=219.ckpt` (best val_loss)
- `fold_2/last.ckpt`, `fold_2/epoch=209.ckpt` (best val_loss)

## Script
`experiments/scripts/train_cv.py`
