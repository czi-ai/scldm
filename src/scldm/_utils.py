import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable, Sequence

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy import sparse

from scldm.constants import ModelEnum
from scldm.logger import logger


def _iter_adatas(adata_or_paths: ad.AnnData | str | Path | Sequence[str | Path]) -> Iterable[ad.AnnData]:
    if isinstance(adata_or_paths, ad.AnnData):
        yield adata_or_paths
        return
    if isinstance(adata_or_paths, (str, Path)):
        yield ad.read_h5ad(adata_or_paths)
        return
    if isinstance(adata_or_paths, Sequence):
        for path in adata_or_paths:
            yield ad.read_h5ad(path)
        return
    raise TypeError("adata_or_paths must be AnnData, a path, or a sequence of paths")


def _compute_log_library_sizes(adata: ad.AnnData) -> np.ndarray:
    counts = adata.X
    if sparse.issparse(counts):
        library_sizes = np.asarray(counts.sum(axis=1)).reshape(-1)
    else:
        library_sizes = np.asarray(counts.sum(axis=1)).reshape(-1)
    return np.log(library_sizes)


def _accumulate_group_stats(
    stats_acc: dict[str, list[float]],
    keys: np.ndarray,
    values: np.ndarray,
) -> None:
    df = pd.DataFrame({"key": keys, "log_ls": values})
    grouped = df.groupby("key")["log_ls"]
    counts = grouped.count()
    sums = grouped.sum()
    sumsq = grouped.apply(lambda s: np.square(s).sum())
    for key in counts.index:
        count = float(counts.loc[key])
        sum_val = float(sums.loc[key])
        sumsq_val = float(sumsq.loc[key])
        if key not in stats_acc:
            stats_acc[key] = [0.0, 0.0, 0.0]
        stats_acc[key][0] += count
        stats_acc[key][1] += sum_val
        stats_acc[key][2] += sumsq_val


def _finalize_stats(stats_acc: dict[str, list[float]]) -> dict[str, float]:
    stats: dict[str, float] = {}
    for key, (count, sum_val, sumsq_val) in stats_acc.items():
        if count <= 0:
            continue
        mean_val = sum_val / count
        stats[key] = float(mean_val)
    return stats


def _finalize_std(stats_acc: dict[str, list[float]]) -> dict[str, float]:
    stats: dict[str, float] = {}
    for key, (count, sum_val, sumsq_val) in stats_acc.items():
        if count <= 0:
            continue
        mean_val = sum_val / count
        var_val = max((sumsq_val / count) - (mean_val**2), 0.0)
        stats[key] = float(np.sqrt(var_val))
    return stats


def compute_log_size_factor_stats(
    adata_or_paths: ad.AnnData | str | Path | Sequence[str | Path],
    class_keys: Sequence[str],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Compute log size factor mean/std per class or class-pair from AnnData."""
    if len(class_keys) not in (1, 2):
        raise ValueError("class_keys must contain 1 or 2 columns")

    class_keys = list(class_keys)
    unique_class1: set[str] = set()
    unique_class2: set[str] = set()
    acc_class1: dict[str, list[float]] = {}
    acc_combo: dict[str, list[float]] = {}

    for adata in _iter_adatas(adata_or_paths):
        for key in class_keys:
            if key not in adata.obs:
                raise ValueError(f"'{key}' column not found in adata.obs")

        log_library_sizes = _compute_log_library_sizes(adata)
        class1_vals = adata.obs[class_keys[0]].astype(str).to_numpy()
        unique_class1.update(np.unique(class1_vals).tolist())
        _accumulate_group_stats(acc_class1, class1_vals, log_library_sizes)

        if len(class_keys) == 2:
            class2_vals = adata.obs[class_keys[1]].astype(str).to_numpy()
            unique_class2.update(np.unique(class2_vals).tolist())
            combo_vals = (class1_vals + "_" + class2_vals).astype(str)
            _accumulate_group_stats(acc_combo, combo_vals, log_library_sizes)

    mu_stats: dict[str, dict[str, float]] = {}
    sd_stats: dict[str, dict[str, float]] = {}

    if len(class_keys) == 1:
        label = class_keys[0]
        mu_stats[label] = _finalize_stats(acc_class1)
        sd_stats[label] = _finalize_std(acc_class1)
        return mu_stats, sd_stats

    joint_key = "_".join(class_keys)
    mu_class1 = _finalize_stats(acc_class1)
    sd_class1 = _finalize_std(acc_class1)
    mu_combo = _finalize_stats(acc_combo)
    sd_combo = _finalize_std(acc_combo)

    mu_stats[joint_key] = {}
    sd_stats[joint_key] = {}
    for class1 in sorted(unique_class1):
        if class1 not in mu_class1 or class1 not in sd_class1:
            raise ValueError(f"Missing size-factor stats for primary class '{class1}'")
        for class2 in sorted(unique_class2):
            combo_key = f"{class1}_{class2}"
            if combo_key in mu_combo and combo_key in sd_combo:
                mu_stats[joint_key][combo_key] = mu_combo[combo_key]
                sd_stats[joint_key][combo_key] = sd_combo[combo_key]
            else:
                mu_stats[joint_key][combo_key] = mu_class1[class1]
                sd_stats[joint_key][combo_key] = sd_class1[class1]

    return mu_stats, sd_stats


def wsd_schedule(
    num_training_steps,
    final_lr_factor=0.1,
    num_warmup_steps=1000,
    init_div_factor=100,
    fract_decay=0.1,
    decay_type="cosine",
):
    """Warmup, hold, and decay schedule.

    Args:
        n_iterations: total number of iterations
        final_lr_factor: factor by which to reduce max_lr at the end
        num_warmup_steps: fraction of iterations used for warmup
        init_div_factor: initial division factor for warmup
        fract_decay: fraction of iterations used for decay
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate
    """
    n_anneal_steps = int(fract_decay * num_training_steps)
    n_hold = num_training_steps - n_anneal_steps

    def schedule(step):
        if step < num_warmup_steps:
            return (step / num_warmup_steps) + (1 - step / num_warmup_steps) / init_div_factor
        elif step < n_hold:
            return 1.0
        elif step < num_training_steps:
            if decay_type == "cosine":
                # Implement cosine decay from warmup to end
                decay_progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
                return final_lr_factor + (1 - final_lr_factor) * 0.5 * (1 + math.cos(math.pi * decay_progress))
            elif decay_type == "sqrt":
                return final_lr_factor + (1 - final_lr_factor) * (1 - math.sqrt((step - n_hold) / n_anneal_steps))
            else:
                raise ValueError(f"decay type {decay_type} is not in ['cosine','sqrt']")
        else:
            return final_lr_factor

    return schedule


def setup_datamodule_and_steps(cfg: DictConfig, world_size: int, num_epochs: int):
    """Setup datamodule and compute training steps.

    This function instantiates the datamodule, computes the effective number of
    training steps based on dataset size and batch size, and updates the config
    accordingly.

    Parameters
    ----------
    cfg
        Hydra configuration object
    world_size
        Number of GPUs/processes for distributed training
    num_epochs
        Number of training epochs

    Returns
    -------
    datamodule
        Instantiated datamodule
    """
    datamodule = hydra.utils.instantiate(cfg.datamodule.datamodule)
    n_cells = datamodule.n_cells
    batch_size = cfg.model.batch_size

    num_steps_per_epoch = n_cells // (batch_size * world_size)
    effective_steps = num_epochs * num_steps_per_epoch

    logger.info(f"Dataset size: {n_cells} cells")
    logger.info(f"Batch size: {batch_size}, World size: {world_size}")
    logger.info(f"Steps per epoch: {num_steps_per_epoch}")
    logger.info(f"Total training steps: {effective_steps}")

    cfg.training.trainer.max_steps = effective_steps

    # Update scheduler warmup (10% of total steps)
    warmup_steps = int(0.1 * effective_steps)

    if hasattr(cfg.model.module, "vae_scheduler") and cfg.model.module.vae_scheduler is not None:
        cfg.model.module.vae_scheduler.num_warmup_steps = warmup_steps
        logger.info(f"VAE warmup steps: {warmup_steps}")

    if hasattr(cfg.model.module, "diffusion_scheduler") and cfg.model.module.diffusion_scheduler is not None:
        cfg.model.module.diffusion_scheduler.num_warmup_steps = warmup_steps
        logger.info(f"Diffusion warmup steps: {warmup_steps}")

    return datamodule


def sort_h5ad_files(path: Path) -> list[str]:
    return sorted(
        [file.as_posix() for file in path.glob("*.h5ad")],
        key=lambda x: int(x.replace(".h5ad", "").split("_")[-1]),
    )


def get_tissue_adata_files(base_path: Path, split: str = "train") -> tuple[list[str], int, int]:
    base_path = Path(base_path)
    all_files = []
    shard_size = []
    total_cells = 0

    for tissue_dir in base_path.iterdir():
        if tissue_dir.is_dir() and "genes" not in str(tissue_dir):
            split_dir = tissue_dir / split
            if split_dir.exists():
                # Read metadata file
                metadata_file = split_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        # Add cells excluding last shard
                        total_cells += metadata["n_cells"] - metadata["last_shard_size"]
                    shard_size.append(metadata["shard_size"])

                h5ad_files = sort_h5ad_files(split_dir)

                # Remove the last file (highest numbered)
                if h5ad_files:
                    all_files.extend(h5ad_files[:-1])

    shard_size_set = set(shard_size)
    assert len(shard_size_set) == 1, "shard_size mismatch"

    return sorted(all_files), total_cells, shard_size_set.pop()


def remap_config(cfg):
    """Recursively remap scg_vae references to scldm in config."""
    if isinstance(cfg, DictConfig):
        for key, value in cfg.items():
            if isinstance(value, str) and "scg_vae" in value:
                cfg[key] = value.replace("scg_vae", "scldm")
            elif isinstance(value, (DictConfig, dict)):
                remap_config(value)
    elif isinstance(cfg, dict):
        for key, value in cfg.items():
            if isinstance(value, str) and "scg_vae" in value:
                cfg[key] = value.replace("scg_vae", "scldm")
            elif isinstance(value, (DictConfig, dict)):
                remap_config(value)


class RemapUnpickler(pickle.Unpickler):
    """Unpickler that remaps scg_vae module names to scldm."""

    def find_class(self, module, name):
        if module.startswith("scg_vae"):
            module = module.replace("scg_vae", "scldm")
        return super().find_class(module, name)


class RemapPickle:
    """Pickle-compatible module for remapping scg_vae to scldm during unpickling."""

    __name__ = "remap_pickle"
    Unpickler = RemapUnpickler
    load = staticmethod(pickle.load)
    dump = staticmethod(pickle.dump)


remap_pickle = RemapPickle()


def process_generation_output(
    output: list[dict[str, torch.Tensor]],
    datamodule: Any,
) -> ad.AnnData:
    logger.info("Processing generation output")
    # counts_true_sparse = sparse.vstack([sparse.csr_matrix(o[f"{ModelEnum.COUNTS.value}"].numpy()) for o in output])
    counts_generated_unconditional_sparse = sparse.vstack(
        [sparse.csr_matrix(o[f"{ModelEnum.COUNTS.value}_generated_unconditional"].numpy()) for o in output]
    )
    counts_generated_conditional_sparse = sparse.vstack(
        [sparse.csr_matrix(o[f"{ModelEnum.COUNTS.value}_generated_conditional"].numpy()) for o in output]
    )
    z_generated_unconditional = np.vstack([o["z_generated_unconditional"].numpy() for o in output])
    z_generated_conditional = np.vstack([o["z_generated_conditional"].numpy() for o in output])

    genes = output[0][f"{ModelEnum.GENES.value}"][0, :]
    var_names = datamodule.vocabulary_encoder.decode_genes(genes)

    # Only decode labels that are present in the output; skip missing ones
    available_keys = set.intersection(*[set(o.keys()) for o in output]) if output else set()
    desired_keys = set(datamodule.vocabulary_encoder.labels.keys())
    present_label_keys = sorted(desired_keys & available_keys)
    missing_label_keys = sorted(desired_keys - available_keys)
    if missing_label_keys:
        logger.info(f"[generation_output] Skipping missing label columns in outputs: {missing_label_keys}")

    # Stack label tensors/arrays robustly, then decode
    obs = {}
    for k in present_label_keys:
        parts = []
        for o in output:
            v = o[k]
            if torch.is_tensor(v):
                parts.append(v.detach().cpu().numpy())
            else:
                parts.append(np.asarray(v))
        stacked = np.concatenate(parts, axis=0)
        obs[k] = datamodule.vocabulary_encoder.decode_metadata(stacked, k)

    del output

    n_cells = counts_generated_unconditional_sparse.shape[0]

    obs_generated_unconditional = pd.DataFrame(obs, index=np.arange(n_cells).astype(str))
    obs_generated_conditional = pd.DataFrame(obs, index=np.arange(n_cells, 2 * n_cells).astype(str))

    obs_generated_unconditional["dataset"] = "generated_unconditional"
    obs_generated_conditional["dataset"] = "generated_conditional"

    X_combined = sparse.vstack([counts_generated_unconditional_sparse, counts_generated_conditional_sparse])
    z_combined = np.vstack([z_generated_unconditional, z_generated_conditional])

    obs_combined = pd.concat([obs_generated_unconditional, obs_generated_conditional], axis=0)
    adata = ad.AnnData(X=X_combined, obs=obs_combined, obsm={"z": z_combined})
    adata.var_names = var_names
    return adata


def create_anndata_from_inference_output(
    output: dict[str, torch.Tensor],
    datamodule: Any,
) -> ad.AnnData:
    obsm: dict[str, np.ndarray] = {}
    if "z" in output:
        obsm["z"] = output["z"].numpy()
    if "z_mean_flat" in output:
        obsm["z_mean_flat"] = output["z_mean_flat"].numpy()
    if "z_sample_flat" in output:
        obsm["z_sample_flat"] = output["z_sample_flat"].numpy()
    if not obsm:
        raise KeyError(f"Missing latent keys in inference output. Keys: {sorted(output.keys())}")

    if "reconstructed_counts" in output:
        generated_counts = sparse.csr_matrix(output["reconstructed_counts"].numpy())
    elif ModelEnum.COUNTS.value in output:
        generated_counts = sparse.csr_matrix(output[ModelEnum.COUNTS.value].numpy())
    else:
        raise KeyError(f"Missing counts in inference output. Keys: {sorted(output.keys())}")

    if ModelEnum.GENES.value not in output:
        raise KeyError(f"Missing genes in inference output. Keys: {sorted(output.keys())}")
    genes = output[ModelEnum.GENES.value][0, :]
    var_names = datamodule.vocabulary_encoder.decode_genes(genes)
    if datamodule.vocabulary_encoder.labels is not None:
        obs = {
            k: datamodule.vocabulary_encoder.decode_metadata(output[k].numpy(), k)
            for k in datamodule.vocabulary_encoder.labels.keys()
        }
    else:
        obs = {}

    n_cells = generated_counts.shape[0]
    obs = pd.DataFrame(obs, index=np.arange(n_cells).astype(str))

    adata = ad.AnnData(
        X=generated_counts,
        obs=obs,
        obsm=obsm,
    )
    adata.var_names = var_names
    adata.layers["counts"] = adata.X.copy()
    return adata


def process_inference_output(
    output: list[ad.AnnData | dict[str, Any]],
    datamodule: Any,
) -> ad.AnnData:
    logger.info("Processing inference output")

    # Flatten nested lists (PyTorch Lightning may return [[AnnData], [AnnData], ...])
    def flatten_output(outputs):
        flattened = []
        for item in outputs:
            if isinstance(item, list):
                flattened.extend(flatten_output(item))
            elif isinstance(item, ad.AnnData):
                flattened.append(item)
            elif isinstance(item, dict):
                flattened.append(item)
            elif item is not None:
                logger.warning(f"Unexpected output type in predict results: {type(item)}. Skipping.")
        return flattened

    def collect_output_types(outputs):
        types = []
        for item in outputs:
            if isinstance(item, list):
                types.extend(collect_output_types(item))
            else:
                types.append(type(item).__name__)
        return types

    flattened_output = flatten_output(output)

    if not flattened_output:
        output_types = collect_output_types(output)
        raise ValueError(f"No valid AnnData objects found in prediction output. Observed output types: {output_types}")

    if all(isinstance(item, dict) for item in flattened_output):
        if any(f"{ModelEnum.COUNTS.value}_generated_unconditional" in item for item in flattened_output):
            return process_generation_output(flattened_output, datamodule)
        flattened_output = [create_anndata_from_inference_output(item, datamodule) for item in flattened_output]

    logger.info(f"Concatenating {len(flattened_output)} AnnData objects")
    adata = ad.concat(flattened_output)

    return adata


def load_validate_statedict_config(
    checkpoints: dict[str, Any],
    config: DictConfig,
    pretrain_config: DictConfig,
) -> tuple[dict[str, Any], DictConfig]:
    """Load VAE state dict from checkpoint and update config with pretrained VAE architecture.

    This function extracts the VAE state dict from a checkpoint (stripping the "vae_model."
    prefix), copies the VAE model config from the pretrained config to the current config,
    and sets the diffusion model input dimensions from the VAE latent space.

    Parameters
    ----------
    checkpoints
        Checkpoint dictionary containing "state_dict" key
    config
        Current configuration to update
    pretrain_config
        Configuration from the pretrained VAE checkpoint

    Returns
    -------
    vae_state_dict
        VAE state dictionary with "vae_model." prefix stripped
    config
        Updated configuration with VAE architecture from pretrained config
    """
    vae_state_dict = {
        k.replace("vae_model.", "", 1): v for k, v in checkpoints["state_dict"].items() if k.startswith("vae_model.")
    }
    config.model.module.vae_model = pretrain_config.model.module.vae_model
    config.model.module.diffusion_model.n_embed_input = pretrain_config.model.module.vae_model.encoder.n_embed_latent
    config.model.module.diffusion_model.seq_len = pretrain_config.model.module.vae_model.encoder.n_inducing_points
    config.model.decoder_name = pretrain_config.model.decoder_name
    return vae_state_dict, config
