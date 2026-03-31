"""3-fold CV training of ScviVAE (NB likelihood) on dentate_gyrus."""

import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import KFold
from torch.distributions import Normal

from scldm._utils import wsd_schedule
from scldm.datamodule import DataModule
from scldm.encoder import VocabularyEncoderSimplified
from scldm.models import VAEScvi
from scldm.nnets import DecoderScvi, EncoderScvi
from scldm.optimizers import AdamWLegacy
from scldm.stochastic_layers import GaussianLinearLayer, NegativeBinomialLinearLayer
from scldm.vae import ScviVAE

# --- Paths ---
REPO_ROOT = Path("/hpc/mydata/giovanni.palla/repos/scldm")
TRAIN_H5AD = REPO_ROOT / "_artifacts/datasets/dentategyrus_train.h5ad"
CKPT_BASE = REPO_ROOT / ".claude/worktrees/bold-ray-jlwk/experiments/checkpoints/scvi_dentate_gyrus_cv"

# --- Hyperparams ---
N_GENES = 17002
N_HIDDEN = 2048
N_LATENT = 2048
N_LAYERS = 2
DROPOUT = 0.1
GENES_SEQ_LEN = 6147
BATCH_SIZE = 128
LR = 1e-3
NUM_EPOCHS = 500
SEED = 12345
N_FOLDS = 3


def compute_size_factors(adata: ad.AnnData, label_col: str = "clusters"):
    """Compute per-cluster log size factor mu and sd."""
    log_lib = np.log(np.array(adata.X.sum(axis=1)).flatten())
    mu_dict = {}
    sd_dict = {}
    for cluster in adata.obs[label_col].cat.categories:
        mask = adata.obs[label_col] == cluster
        vals = log_lib[mask]
        mu_dict[cluster] = float(np.mean(vals))
        sd_dict[cluster] = float(np.std(vals))
    return {label_col: mu_dict}, {label_col: sd_dict}


def build_model(num_training_steps: int) -> VAEScvi:
    encoder = EncoderScvi(
        n_genes=N_GENES,
        n_hidden=N_HIDDEN,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    )
    encoder_head = GaussianLinearLayer(
        n_hidden=N_HIDDEN,
        n_latent=N_LATENT,
    )
    decoder = DecoderScvi(
        n_latent=N_LATENT,
        n_hidden=N_HIDDEN,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    )
    decoder_head = NegativeBinomialLinearLayer(
        n_genes=N_GENES,
        n_hidden=N_HIDDEN,
        shared_theta=True,
    )
    # Prior will be created on the correct device via a wrapper
    prior = Normal(torch.zeros(N_LATENT), torch.ones(N_LATENT))

    vae_model = ScviVAE(
        encoder=encoder,
        encoder_head=encoder_head,
        decoder=decoder,
        decoder_head=decoder_head,
        prior=prior,
    )
    # Register prior params as buffers so they move to GPU with the model
    vae_model.register_buffer("_prior_loc", torch.zeros(N_LATENT))
    vae_model.register_buffer("_prior_scale", torch.ones(N_LATENT))

    # Patch the prior property to use device-aware buffers
    _original_forward = vae_model.forward

    def _patched_forward(counts, genes, library_size, condition=None, counts_subset=None, genes_subset=None, masking_prop=0.0, mask_token_idx=0):
        vae_model.prior = Normal(vae_model._prior_loc, vae_model._prior_scale)
        return _original_forward(counts, genes, library_size, condition, counts_subset, genes_subset, masking_prop, mask_token_idx)

    vae_model.forward = _patched_forward

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

    return VAEScvi(
        vae_model=vae_model,
        vae_optimizer=vae_optimizer,
        vae_scheduler=vae_scheduler,
        kl_weight=1.0,
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

    # Save fold splits and size factors
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
