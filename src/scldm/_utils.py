import json
import math
import pickle
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy import sparse

from scldm.constants import ModelEnum
from scldm.logger import logger


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
                # print(split_dir)
                # print(h5ad_files[0])
                # print(h5ad_files[-1])

                # Remove the last file (highest numbered)
                if h5ad_files:
                    all_files.extend(h5ad_files[:-1])

    shard_size = set(shard_size)
    assert len(shard_size) == 1, "shard_size mismatch"

    return sorted(all_files), total_cells, shard_size.pop()


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
    z = output["z"].numpy()
    generated_counts = sparse.csr_matrix(output["reconstructed_counts"].numpy())
    genes = output[f"{ModelEnum.GENES.value}"][0, :]
    var_names = datamodule.vocabulary_encoder.decode_genes(genes)
    if datamodule.vocabulary_encoder.labels is not None:
        obs = {
            k: datamodule.vocabulary_encoder.decode_metadata(output[k].numpy(), k)
            for k in datamodule.vocabulary_encoder.labels.keys()
        }
    else:
        obs = {}

    n_cells = len(z)
    obs = pd.DataFrame(obs, index=np.arange(n_cells).astype(str))

    adata = ad.AnnData(
        X=generated_counts,
        obs=obs,
        obsm={
            "z": z,
        },
    )
    adata.var_names = var_names
    adata.layers["counts"] = adata.X.copy()
    return adata


def process_inference_output(
    output: list[ad.AnnData],
    datamodule: Any,
) -> ad.AnnData:
    logger.info("Processing inference output")
    adata = ad.concat(output)

    return adata
