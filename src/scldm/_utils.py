import json
import math
import os
import pickle
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from torch.utils._pytree import tree_map
from torch.utils.flop_counter import FlopCounterMode


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr: float
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


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


class MaskingSchedulerCallback(Callback):
    def __init__(
        self,
        start_proportion,
        end_proportion,
        total_steps,
        schedule_type="linear",
    ):
        self.start_proportion = start_proportion
        self.end_proportion = end_proportion
        self.total_steps = total_steps
        self.current_proportion = start_proportion
        self.schedule_type = schedule_type
        self.betalinear30_dist = torch.distributions.Beta(torch.tensor(3.0), torch.tensor(9.0))
        self.uniform_dist = torch.distributions.uniform.Uniform(0, 1)

    def _get_betalinear30_sample(self):
        if self.uniform_dist.sample().item() < 0.8:
            return self.betalinear30_dist.sample().item()
        else:
            return self.uniform_dist.sample().item()

    def _get_linear_sample(self):
        return torch.distributions.uniform.Uniform(0, 1).sample().item()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # global_step = trainer.global_step
        # progress = min(global_step / self.total_steps, 1.0)

        if self.schedule_type == "linear":
            self.current_proportion = self._get_linear_sample()
        elif self.schedule_type == "betalinear30":
            self.current_proportion = self._get_betalinear30_sample()
        else:
            raise ValueError(f"Invalid schedule type: {self.schedule_type}")

        self.current_proportion = max(0.0, min(0.999, self.current_proportion))
        pl_module.mask_proportion = self.current_proportion


def world_info_from_env():
    # from https://github.com/mlfoundations/open_clip/blob/main/src/training/distributed.py
    local_rank = 0
    for v in ("LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


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


def get_flops(
    datamodule: pl.LightningDataModule,
    module: pl.LightningModule,
    with_backward: bool = True,
):
    # not working, see https://github.com/pytorch/pytorch/issues/134385
    datamodule.setup("test")
    dl = datamodule.test_dataloader()
    batch = next(iter(dl))
    batch_ = module.tokens_and_masks(batch)
    batch_.pop("local_non_padding_tokens")
    module.transformer.to("cuda:0")
    module.count_head.to("cuda:0")
    batch_ = tree_map(lambda x: x.to("cuda:0"), batch_)
    batch_ = {k: v[0, ...].unsqueeze(0) for k, v in batch_.items()}

    flop_counter = FlopCounterMode(mods=module, display=False, depth=None)
    with flop_counter:
        if with_backward:
            module(batch_).sum().backward()
        else:
            module(batch_)
    total_flops = flop_counter.get_total_flops()
    return total_flops


def get_inducing_points(n_inducing_points: int):
    n_inducing_points = (
        [n_inducing_points] if isinstance(n_inducing_points, int) else [int(x) for x in n_inducing_points.split("-")]
    )
    return n_inducing_points


def get_n_embed_inducing_points(n_embed: int, n_inducing_points: int):
    n_embed_list = [n_embed * (2**i) for i in range(len(n_inducing_points) + 1)]
    return n_embed_list


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


def process_generation(generation_output: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    import anndata as ad
    from scipy import sparse

    generated_counts = torch.cat(
        [
            generation_output[i][f"{ModelEnum.COUNTS.value}_generated"]
            for i in range(len(generation_output))
            if generation_output[i] is not None
        ]
    )
    true_counts = torch.cat(
        [
            generation_output[i][ModelEnum.COUNTS.value]
            for i in range(len(generation_output))
            if generation_output[i] is not None
        ]
    )

    adata_true = ad.AnnData(sparse.csr_matrix(true_counts.numpy()))
    adata_generated = ad.AnnData(sparse.csr_matrix(generated_counts.numpy()))
    return adata_true, adata_generated


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
    z_sample = output["z_sample"].numpy()
    z_sample_flat = output["z_sample_flat"].numpy()
    generated_counts = sparse.csr_matrix(output["reconstructed_counts"].numpy())
    genes = output[f"{ModelEnum.GENES.value}"][0, :]
    var_names = datamodule.vocabulary_encoder.decode_genes(genes)
    obs = {
        k: datamodule.vocabulary_encoder.decode_metadata(output[k].numpy(), k)
        for k in datamodule.vocabulary_encoder.labels.keys()
    }

    n_cells = len(z_sample)
    obs = pd.DataFrame(obs, index=np.arange(n_cells).astype(str))

    adata = ad.AnnData(X=generated_counts, obs=obs, obsm={"z_sample": z_sample, "z_sample_flat": z_sample_flat})
    adata.var_names = var_names
    adata.layers["counts"] = adata.X.copy()
    return adata


def process_inference_output(
    output: list[ad.AnnData],
    datamodule: Any,
) -> ad.AnnData:
    logger.info("Processing inference output")
    adata = ad.concat(output)

    # sc.pp.normalize_total(adata)
    # sc.pp.log1p(adata)
    # sc.pp.pca(adata)
    # sc.pp.neighbors(adata)
    # sc.tl.umap(adata)

    # # process latents
    # adata.obsm["z_sample_flat_pca"] = sc.pp.pca(adata.obsm["z_sample_flat"])

    # sc.pp.neighbors(adata, use_rep="z_sample", key_added="z_sample_neighbors")
    # sc.pp.neighbors(adata, use_rep="z_sample_flat_pca", key_added="z_sample_flat_pca_neighbors", n_neighbors=10)

    # sc.tl.umap(adata, neighbors_key="z_sample_neighbors", key_added="z_sample_neighbors_umap")
    # sc.tl.umap(adata, neighbors_key="z_sample_flat_pca_neighbors", key_added="z_sample_flat_pca_neighbors_umap")

    return adata
