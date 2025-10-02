import json
import math
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
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
