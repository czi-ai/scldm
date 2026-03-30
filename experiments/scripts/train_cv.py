"""3-fold CV training of TransformerVAE (NB likelihood) on dentate_gyrus."""

import os
import pickle
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import KFold

from scldm._utils import wsd_schedule
from scldm.datamodule import DataModule
from scldm.encoder import VocabularyEncoderSimplified
from scldm.layers import InputTransformerVAE
from scldm.models import VAE
from scldm.nnets import Decoder, Encoder
from scldm.optimizers import AdamWLegacy
from scldm.stochastic_layers import NegativeBinomialTransformerLayer
from scldm.vae import TransformerVAE

# --- Paths ---
REPO_ROOT = Path("/hpc/mydata/giovanni.palla/repos/scldm")
TRAIN_H5AD = REPO_ROOT / "_artifacts/datasets/dentategyrus_train.h5ad"
CKPT_BASE = REPO_ROOT / ".claude/worktrees/bold-ray-jlwk/experiments/checkpoints/vae_dentate_gyrus_cv"
SIZE_FACTOR_MU = REPO_ROOT / "_artifacts/resubmission/dentate_gyrus_log_size_factor_mu.pkl"
SIZE_FACTOR_SD = REPO_ROOT / "_artifacts/resubmission/dentate_gyrus_log_size_factor_sd.pkl"

# --- Hyperparams (matching pre-trained config) ---
N_GENES = 17002
N_INDUCING_POINTS = 128
N_EMBED = 128
N_EMBED_LATENT = 16
N_HEAD = 4
N_HEAD_CROSS = 1
N_LAYER = 2
GENES_SEQ_LEN = 6147
BATCH_SIZE = 128
LR = 1e-3
NUM_EPOCHS = 500
SEED = 12345
N_FOLDS = 3


def compute_size_factors(adata: ad.AnnData, label_col: str = "clusters"):
    """Compute per-cluster log size factor mu and sd, matching the format of the existing pickle files."""
    log_lib = np.log(np.array(adata.X.sum(axis=1)).flatten())
    mu_dict = {}
    sd_dict = {}
    for cluster in adata.obs[label_col].cat.categories:
        mask = adata.obs[label_col] == cluster
        vals = log_lib[mask]
        mu_dict[cluster] = float(np.mean(vals))
        sd_dict[cluster] = float(np.std(vals))
    return {label_col: mu_dict}, {label_col: sd_dict}


def build_model(num_training_steps: int) -> VAE:
    encoder = Encoder(
        n_layer=N_LAYER,
        n_inducing_points=N_INDUCING_POINTS,
        n_embed=N_EMBED,
        n_embed_latent=N_EMBED_LATENT,
        n_head=N_HEAD,
        n_head_cross=N_HEAD_CROSS,
        dropout=0.0,
        bias=False,
        multiple_of=4,
        layernorm_eps=1e-8,
        norm_layer="layernorm",
        positional_encoding=True,
    )
    decoder = Decoder(
        n_genes=N_GENES,
        n_embed=N_EMBED,
        n_embed_latent=N_EMBED_LATENT,
        n_head=N_HEAD,
        n_head_cross=N_HEAD_CROSS,
        n_layer=N_LAYER,
        n_inducing_points=N_INDUCING_POINTS,
        dropout=0.0,
        bias=False,
        multiple_of=4,
        layernorm_eps=1e-8,
        norm_layer="layernorm",
        shared_embedding=True,
        use_adaln=False,
    )
    input_layer = InputTransformerVAE(
        n_genes=N_GENES,
        n_embed=N_EMBED,
        agg_func="log1p",
    )
    decoder_head = NegativeBinomialTransformerLayer(
        n_genes=N_GENES,
        shared_theta=True,
        n_embed=N_EMBED,
        norm_layer="layernorm",
        layernorm_eps=1e-8,
    )
    vae_model = TransformerVAE(
        encoder=encoder,
        decoder=decoder,
        decoder_head=decoder_head,
        input_layer=input_layer,
    )

    warmup_steps = int(0.1 * num_training_steps)
    vae_scheduler = wsd_schedule(
        num_training_steps=num_training_steps,
        final_lr_factor=0.1,
        num_warmup_steps=warmup_steps,
        init_div_factor=100,
        fract_decay=0.1,
        decay_type="sqrt",
    )
    vae_optimizer = lambda params: AdamWLegacy(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95), caution=False)

    return VAE(
        vae_model=vae_model,
        vae_optimizer=vae_optimizer,
        vae_scheduler=vae_scheduler,
        calculate_grad_norms=False,
        compile=False,
    )


def train_fold(fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray, full_adata: ad.AnnData):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx} — train={len(train_idx)}, val={len(val_idx)}")
    print(f"{'='*60}\n")

    pl.seed_everything(SEED + fold_idx)

    fold_ckpt_dir = CKPT_BASE / f"fold_{fold_idx}"
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Split data
    train_adata = full_adata[train_idx].copy()
    val_adata = full_adata[val_idx].copy()

    # Compute size factors from fold training data
    mu_sf, sd_sf = compute_size_factors(train_adata, "clusters")

    # Save fold splits and size factors to temp files
    tmpdir = fold_ckpt_dir / "data"
    tmpdir.mkdir(exist_ok=True)
    train_path = tmpdir / "train.h5ad"
    val_path = tmpdir / "val.h5ad"
    mu_path = tmpdir / "mu_size_factor.pkl"
    sd_path = tmpdir / "sd_size_factor.pkl"

    train_adata.write_h5ad(train_path)
    val_adata.write_h5ad(val_path)
    with open(mu_path, "wb") as f:
        pickle.dump(mu_sf, f)
    with open(sd_path, "wb") as f:
        pickle.dump(sd_sf, f)

    # Build vocabulary encoder
    vocab_encoder = VocabularyEncoderSimplified(
        adata_path=str(train_path),
        class_vocab_sizes={"clusters": 14},
        mask_token="<MASK>",
        mask_token_idx=0,
        n_genes=N_GENES,
        guidance_weight={"clusters": 1.0},
        mu_size_factor=str(mu_path),
        sd_size_factor=str(sd_path),
        condition_strategy="mutually_exclusive",
    )

    # Build datamodule
    datamodule = DataModule(
        train_adata_path=train_path,
        test_adata_path=val_path,
        adata_attr="X",
        adata_key=None,
        vocabulary_encoder=vocab_encoder,
        val_as_test=True,
        batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,
        num_workers=4,
        seed=SEED,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last_indices=True,
        drop_incomplete_batch=True,
        sample_genes="expressed",
        genes_seq_len=GENES_SEQ_LEN,
    )
    datamodule.setup()

    # Compute training steps
    n_cells = datamodule.n_cells
    steps_per_epoch = n_cells // BATCH_SIZE
    total_steps = NUM_EPOCHS * steps_per_epoch
    print(f"Cells: {n_cells}, Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")

    # Build model
    model = build_model(total_steps)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(fold_ckpt_dir),
        filename="{epoch}",
        save_weights_only=False,
        save_on_train_epoch_end=True,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        enable_version_counter=False,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step", log_weight_decay=True)

    # CSV logger
    csv_logger = CSVLogger(
        save_dir=str(fold_ckpt_dir),
        name="csv_logs",
    )

    # Trainer
    trainer = Trainer(
        max_steps=total_steps,
        enable_progress_bar=True,
        precision=32,
        log_every_n_steps=30,
        check_val_every_n_epoch=10,
        accelerator="gpu",
        devices=1,
        strategy="auto",
        enable_checkpointing=True,
        deterministic=False,
        gradient_clip_val=10.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=1,
        logger=csv_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        use_distributed_sampler=False,
    )

    # Resume if checkpoint exists
    last_ckpt = fold_ckpt_dir / "last.ckpt"
    ckpt_path = str(last_ckpt) if last_ckpt.exists() else None
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    print(f"Fold {fold_idx} complete. Best: {checkpoint_cb.best_model_path}")



def main():
    torch.set_float32_matmul_precision("high")

    print("Loading training data...")
    full_adata = ad.read_h5ad(TRAIN_H5AD)
    print(f"Loaded: {full_adata.shape}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(full_adata.n_obs))):
        train_fold(fold_idx, train_idx, val_idx, full_adata)

    print("\nAll folds complete!")


if __name__ == "__main__":
    main()
