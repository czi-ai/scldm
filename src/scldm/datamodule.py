import json
import os
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import anndata as ad
import numpy as np
import torch
from anndata import AnnData
from pytorch_lightning import LightningDataModule

try:
    from cellarium.ml.data import (
        DistributedAnnDataCollection,
        IterableDistributedAnnDataCollectionDataset,
    )
    from cellarium.ml.utilities.data import AnnDataField, convert_to_tensor
except ImportError as e:
    raise ImportError(
        "cellarium-ml is required for DataModule functionality. "
        "If `cellarium-ml>0.0.7` is available on PyPI, install with: pip install cellarium-ml . "
        "Otherwise: pip install 'cellarium-ml @ git+https://github.com/cellarium-ai/cellarium-ml.git'"
    ) from e
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader

from scldm._utils import compute_log_size_factor_stats, get_tissue_adata_files, sort_h5ad_files
from scldm.constants import ModelEnum
from scldm.encoder import VocabularyEncoderSimplified
from scldm.logger import logger


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_adata_path: Path,
        test_adata_path: Path,
        adata_attr: str,
        adata_key: str | None,
        vocabulary_encoder: VocabularyEncoderSimplified,
        val_as_test: bool = True,
        data_path: Path | None = None,
        allow_missing_train: bool = False,
        batch_size: int = 256,
        test_batch_size: int = 256,
        num_workers: int = 4,
        seed: int = 42,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        drop_last_indices: bool = False,
        drop_incomplete_batch: bool = True,
        sample_genes: Literal["random", "weighted", "expressed", "expressed_zero", "none"] = "none",
        genes_seq_len: int = 100,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        assert isinstance(vocabulary_encoder, VocabularyEncoderSimplified)

        self.vocabulary_encoder = vocabulary_encoder
        self.adata_attr = adata_attr
        self.adata_key = adata_key
        self.train_adata_path = Path(train_adata_path) if train_adata_path is not None else None
        self.test_adata_path = Path(test_adata_path) if test_adata_path is not None else None
        self.val_as_test = val_as_test
        self.allow_missing_train = allow_missing_train
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.drop_last_indices = drop_last_indices
        self.drop_incomplete_batch = drop_incomplete_batch
        self.sample_genes = sample_genes
        self.genes_seq_len = genes_seq_len
        self.data_path = data_path

        # this should be done in `setup`, but we need to read the metadata to get the number of cells
        # this won't work in distributed training
        if "adata_0.h5ad" in str(self.train_adata_path):
            # Read metadata from folder
            metadata_path = os.path.join(self.train_adata_path.parent, "metadata.json")
            with open(metadata_path) as f:
                self.train_metadata = json.load(f)
            self.n_cells = self.train_metadata["n_cells"]
        elif self.data_path is not None:
            _, self.n_cells, _ = get_tissue_adata_files(self.data_path, "train")
            self.train_metadata = None
        elif self.train_adata_path is not None:
            if self.train_adata_path.exists():
                train_adata = ad.read_h5ad(self.train_adata_path)
                self.n_cells = train_adata.n_obs
                self.train_metadata = None
            elif self.allow_missing_train:
                logger.info("Train adata path missing; continuing in predict-only mode")
                self.n_cells = 0
                self.train_metadata = None
            else:
                raise FileNotFoundError(f"Train adata path not found: {self.train_adata_path}")
        else:
            logger.info("No train adata path provided, make sure to set up datamodule for inference")
            self.n_cells = 0

        self._adata_inference = None

    @property
    def adata_inference(self):
        return self._adata_inference

    @adata_inference.setter
    def adata_inference(self, adata: AnnData):
        if hasattr(self.vocabulary_encoder, "genes") and self.vocabulary_encoder.genes is not None:
            available_genes = set(adata.var_names)
            required_genes = set(self.vocabulary_encoder.genes)

            missing_genes = required_genes - available_genes
            kept = available_genes & required_genes
            logger.info(f"Filtering genes to encoder vocabulary: kept={len(kept)}, missing={len(missing_genes)}")

            genes = [g for g in adata.var_names if g in required_genes]
            adata = adata[:, genes].copy()
        self._adata_inference = adata

    def get_library_size(self, x: np.ndarray) -> np.ndarray:
        """Compute library size (total counts per cell)."""
        return x.sum(axis=1, keepdims=True)

    def _setup_prediction_only(self):
        """Set up prediction dataset using adata_inference, skipping all training/validation setup."""
        # Recompute labels for predict based on columns actually present in adata_inference
        predict_labels = {}
        if self.vocabulary_encoder.labels is not None and isinstance(self.vocabulary_encoder.labels, dict):
            present_label_keys = [
                label for label in self.vocabulary_encoder.labels.keys() if label in self.adata_inference.obs
            ]
            missing_label_keys = [
                label for label in self.vocabulary_encoder.labels.keys() if label not in self.adata_inference.obs
            ]
            if missing_label_keys:
                logger.info(f"[predict] Skipping missing label columns in adata_inference: {missing_label_keys}")
            predict_labels = {
                label: AnnDataField(
                    attr="obs",
                    key=label,
                    convert_fn=lambda x, label=label: self.vocabulary_encoder.encode_metadata(x, label=label),
                )
                for label in present_label_keys
            }

        gene_tokens_transform = partial(
            tokenize_cells,
            encoder=self.vocabulary_encoder,
        )
        predict_batch_keys = {
            ModelEnum.COUNTS.value: AnnDataField(
                attr=self.adata_attr, key=self.adata_key, convert_fn=lambda x: x.toarray()
            ),
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr=self.adata_attr,
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        var_names=data_dict["var_names"],
                        genes_seq_len=self.genes_seq_len,
                        sample_genes=self.sample_genes,
                    ),
                ),
            ),
            ModelEnum.LIBRARY_SIZE.value: AnnDataField(
                attr=self.adata_attr,
                key=self.adata_key,
                convert_fn=lambda x: self.get_library_size(x.toarray()),
            ),
            **predict_labels,
        }

        dataset = partial(
            IterableDistributedAnnDataCollectionDataset,
            shuffle_seed=self.seed,
            worker_seed=None,
        )

        self.predict_dataset = dataset(
            batch_keys=predict_batch_keys,  # type: ignore
            dadc=self.adata_inference,
            shuffle=False,
            shuffle_seed=False,
            batch_size=self.test_batch_size,
            drop_last_indices=False,
            drop_incomplete_batch=False,
        )

    def _setup_prediction_from_test(self):
        """Set up prediction dataset using test data, skipping all training/validation setup."""
        labels = {}
        if self.vocabulary_encoder.labels is not None and isinstance(self.vocabulary_encoder.labels, dict):
            labels = {
                label: AnnDataField(
                    attr="obs",
                    key=label,
                    convert_fn=lambda x, label=label: self.vocabulary_encoder.encode_metadata(x, label=label),
                )
                for label in self.vocabulary_encoder.labels.keys()
            }

        gene_tokens_transform = partial(
            tokenize_cells,
            encoder=self.vocabulary_encoder,
        )
        test_batch_keys = {
            ModelEnum.COUNTS.value: AnnDataField(
                attr=self.adata_attr, key=self.adata_key, convert_fn=lambda x: x.toarray()
            ),
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr=self.adata_attr,
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        var_names=data_dict["var_names"],
                        genes_seq_len=self.genes_seq_len,
                        sample_genes=self.sample_genes,
                    ),
                ),
            ),
            ModelEnum.LIBRARY_SIZE.value: AnnDataField(
                attr=self.adata_attr,
                key=self.adata_key,
                convert_fn=lambda x: self.get_library_size(x.toarray()),
            ),
            **labels,
        }

        dataset = partial(
            IterableDistributedAnnDataCollectionDataset,
            shuffle_seed=self.seed,
            worker_seed=None,
        )

        self.predict_dataset = dataset(
            batch_keys=test_batch_keys,  # type: ignore
            dadc=self.test_adata,
            shuffle=False,
            shuffle_seed=False,
            batch_size=self.test_batch_size,
            drop_last_indices=False,
            drop_incomplete_batch=False,
        )

    def setup(self, stage: str | None = None):
        if stage == "predict":
            if self.adata_inference is not None:
                if not isinstance(self.adata_inference, AnnData):
                    raise TypeError("adata_inference must be an AnnData object")
                self._setup_prediction_only()
                return
            else:
                if self.test_adata_path is None:
                    raise ValueError("test_adata_path must be set for predict when adata_inference is not provided")
                if not hasattr(self, "test_adata") or self.test_adata is None:
                    self.test_adata = ad.read_h5ad(self.test_adata_path)
                self._setup_prediction_from_test()
                return

        if "adata_0.h5ad" in str(self.train_adata_path):
            logger.info("Using train_val_split_list from sharded train files")
            # Read metadata from folder
            train_metadata_path = os.path.join(self.train_adata_path.parent, "metadata.json")
            with open(train_metadata_path) as f:
                self.train_metadata = json.load(f)
            test_metadata_path = os.path.join(self.test_adata_path.parent, "metadata.json")
            with open(test_metadata_path) as f:
                self.test_metadata = json.load(f)
            self.train_files = sort_h5ad_files(self.train_adata_path.parent)
            self.test_files = sort_h5ad_files(self.test_adata_path.parent)
        elif self.data_path is not None:
            self.train_files, n_cells_train, shard_size_train = get_tissue_adata_files(self.data_path, "train")
            self.test_files, n_cells_val, shard_size_val = get_tissue_adata_files(self.data_path, "test")
            self.train_metadata = {
                "n_cells": n_cells_train,
                "shard_size": shard_size_train,
                "last_shard_size": shard_size_train,
            }
            self.test_metadata = {
                "n_cells": n_cells_val,
                "shard_size": shard_size_val,
                "last_shard_size": shard_size_val,
            }
        else:
            self.train_adata = ad.read_h5ad(self.train_adata_path)
            self.train_metadata = None
            self.test_metadata = None
            self.test_adata = ad.read_h5ad(self.test_adata_path)

        if self.val_as_test:
            if self.train_metadata is None:
                self.val_adata = self.test_adata
                self.val_ann_collection = self.val_adata
                self.train_ann_collection = self.train_adata
                self.test_ann_collection = self.test_adata

            else:
                self.train_ann_collection = DistributedAnnDataCollection(
                    self.train_files,
                    shard_size=self.train_metadata["shard_size"],
                    last_shard_size=self.train_metadata["last_shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )
                self.val_ann_collection = DistributedAnnDataCollection(
                    self.test_files,
                    shard_size=self.test_metadata["shard_size"],
                    last_shard_size=self.test_metadata["last_shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )
                self.test_ann_collection = DistributedAnnDataCollection(
                    self.test_files,
                    shard_size=self.test_metadata["shard_size"],
                    last_shard_size=self.test_metadata["last_shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )
        else:
            # Split train files into train and validation sets
            if self.train_metadata is None:
                rng = np.random.RandomState(self.seed)
                n_cells = self.train_adata.n_obs
                n_val_cells = int(0.1 * n_cells)
                indices = np.arange(n_cells)
                resample_indices = rng.permutation(indices)

                train_indices = resample_indices[:-n_val_cells]
                val_indices = resample_indices[-n_val_cells:]

                self.val_adata = self.train_adata[val_indices]
                self.train_adata = self.train_adata[train_indices]
                self.train_ann_collection = self.train_adata
                self.val_ann_collection = self.val_adata
                self.test_ann_collection = self.test_adata
            else:
                logger.info("Using train_val_split_list from sharded train files")
                train_indices, val_indices = train_val_split_list(self.train_files, self.seed)
                self.val_files = [self.train_files[i] for i in val_indices]
                self.train_files = [self.train_files[i] for i in train_indices]
                self.train_ann_collection = DistributedAnnDataCollection(
                    self.train_files,
                    shard_size=self.train_metadata["shard_size"],
                    last_shard_size=self.train_metadata["last_shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )
                self.val_ann_collection = DistributedAnnDataCollection(
                    self.val_files,
                    shard_size=self.train_metadata["shard_size"],
                    last_shard_size=self.test_metadata["last_shard_size"]
                    if self.val_as_test
                    else self.train_metadata["shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )
                self.test_ann_collection = DistributedAnnDataCollection(
                    self.test_files,
                    shard_size=self.test_metadata["shard_size"],
                    last_shard_size=self.test_metadata["last_shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )

        if self.vocabulary_encoder.mu_size_factor is None and self.vocabulary_encoder.sd_size_factor is None:
            class_keys = list(self.vocabulary_encoder.class_vocab_sizes.keys())
            if len(class_keys) not in (1, 2):
                raise ValueError("Auto size-factor computation supports only 1 or 2 class columns")
            logger.info(f"Computing size-factor stats from training data using: {class_keys}")
            if self.train_metadata is None:
                mu_stats, sd_stats = compute_log_size_factor_stats(self.train_adata, class_keys)
            else:
                mu_stats, sd_stats = compute_log_size_factor_stats(self.train_files, class_keys)
            self.vocabulary_encoder.set_size_factor_stats(mu_stats, sd_stats)

        labels = {}
        if self.vocabulary_encoder.labels is not None and isinstance(self.vocabulary_encoder.labels, dict):
            labels = {
                label: AnnDataField(
                    attr="obs",
                    key=label,
                    convert_fn=lambda x, label=label: self.vocabulary_encoder.encode_metadata(x, label=label),
                )
                for label in self.vocabulary_encoder.labels.keys()
            }

        gene_tokens_transform = partial(
            tokenize_cells,
            encoder=self.vocabulary_encoder,
        )

        train_batch_keys = {
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr=self.adata_attr,
                key=self.adata_key,
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        genes_seq_len=self.genes_seq_len,
                        var_names=data_dict["var_names"],
                        sample_genes=self.sample_genes,
                    ),
                ),
            ),
            **labels,
        }
        val_batch_keys = {
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr=self.adata_attr,
                key=self.adata_key,
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        genes_seq_len=self.genes_seq_len,
                        var_names=data_dict["var_names"],
                        sample_genes=self.sample_genes,
                    ),
                ),
            ),
            **labels,
        }
        test_batch_keys = {
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr=self.adata_attr,
                key=self.adata_key,
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        genes_seq_len=self.genes_seq_len,
                        var_names=data_dict["var_names"],
                        sample_genes=self.sample_genes,
                    ),
                ),
            ),
            **labels,
        }
        logger.info("Using IterableDistributedAnnDataCollectionDataset", stacklevel=2)

        dataset = partial(
            IterableDistributedAnnDataCollectionDataset,
            shuffle_seed=self.seed,
            worker_seed=None,
        )

        self.train_dataset = dataset(
            batch_keys=train_batch_keys,  # type: ignore
            dadc=self.train_ann_collection,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last_indices=True,
            drop_incomplete_batch=True,
        )
        self.val_dataset = dataset(
            batch_keys=val_batch_keys,  # type: ignore
            dadc=self.val_ann_collection,
            shuffle=False,
            shuffle_seed=False,
            batch_size=self.test_batch_size,
            drop_last_indices=True,
            drop_incomplete_batch=True,
        )
        self.test_dataset = dataset(
            batch_keys=test_batch_keys,  # type: ignore
            dadc=self.test_ann_collection,
            shuffle=False,
            shuffle_seed=False,
            batch_size=self.test_batch_size,
            drop_last_indices=False,
            drop_incomplete_batch=False,
        )
        if stage == "predict":
            if self.adata_inference is not None:
                if not isinstance(self.adata_inference, AnnData):
                    raise TypeError("adata_inference must be an AnnData object")
                # Recompute labels for predict based on columns actually present in adata_inference
                predict_labels = {}
                if self.vocabulary_encoder.labels is not None and isinstance(self.vocabulary_encoder.labels, dict):
                    present_label_keys = [
                        label for label in self.vocabulary_encoder.labels.keys() if label in self.adata_inference.obs
                    ]
                    missing_label_keys = [
                        label
                        for label in self.vocabulary_encoder.labels.keys()
                        if label not in self.adata_inference.obs
                    ]
                    if missing_label_keys:
                        logger.info(
                            f"[predict] Skipping missing label columns in adata_inference: {missing_label_keys}"
                        )
                    predict_labels = {
                        label: AnnDataField(
                            attr="obs",
                            key=label,
                            convert_fn=lambda x, label=label: self.vocabulary_encoder.encode_metadata(x, label=label),
                        )
                        for label in present_label_keys
                    }

                predict_batch_keys = {
                    ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                        attr="X",
                        convert_fn=cast(
                            Any,
                            lambda data_dict: gene_tokens_transform(
                                cell=data_dict["data"],
                                genes_seq_len=self.genes_seq_len,
                                var_names=data_dict["var_names"],
                                sample_genes="none",
                            ),
                        ),
                    ),
                    # propagate original obs_names for alignment downstream
                    "obs_names": AnnDataField(
                        attr="obs_names",
                        convert_fn=lambda x: np.asarray(x),
                    ),
                    **predict_labels,
                }
                self.predict_dataset = dataset(
                    batch_keys=predict_batch_keys,  # type: ignore
                    dadc=self.adata_inference,
                    shuffle=False,
                    shuffle_seed=False,
                    batch_size=self.batch_size,
                    drop_last_indices=False,
                    drop_incomplete_batch=False,
                )
            else:
                predict_batch_keys = deepcopy(test_batch_keys)
                self.predict_dataset = dataset(
                    batch_keys=predict_batch_keys,  # type: ignore
                    dadc=self.test_ann_collection,
                    shuffle=False,
                    shuffle_seed=False,
                    batch_size=self.batch_size,
                    drop_last_indices=False,
                    drop_incomplete_batch=False,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def collate_fn_annloader(
        self,
        batch,
        sample_genes: Literal["random", "weighted", "expressed", "expressed_zero", "none"],
        genes_seq_len: int,
    ):
        output = tokenize_cells(batch.X, batch.var_names, self.vocabulary_encoder, genes_seq_len, sample_genes)
        output.update(
            {
                k: self.vocabulary_encoder.encode_metadata(batch.obs[k].values, label=k)
                for k in self.vocabulary_encoder.labels
            }
        )
        output = tree_map(lambda x: x.detach().clone() if torch.is_tensor(x) else torch.tensor(x), output)
        return output


def collate_fn(
    batch: list[dict[str, dict[str, np.ndarray] | np.ndarray]],
) -> dict[str, dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor]:
    keys = batch[0].keys()
    collated_batch: dict[str, dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor] = {}
    if len(batch) > 1 and not all(keys == data.keys() for data in batch[1:]):
        raise ValueError("All dictionaries in the batch must have the same keys.")
    for key in keys:
        if key == ModelEnum.GENES.value:
            collated_batch[ModelEnum.COUNTS.value] = np.concatenate(
                [data[key][ModelEnum.COUNTS.value] for data in batch],
                axis=0,  # type: ignore
            )
            collated_batch[ModelEnum.GENES.value] = np.concatenate(
                [data[key][ModelEnum.GENES.value] for data in batch],
                axis=0,  # type: ignore
            )
            collated_batch[ModelEnum.LIBRARY_SIZE.value] = np.concatenate(
                [data[key][ModelEnum.LIBRARY_SIZE.value] for data in batch],
                axis=0,  # type: ignore
            )
            # Optional extras if provided by tokenizer
            if ModelEnum.GENES_SUBSET.value in batch[0][key]:  # type: ignore
                collated_batch[ModelEnum.GENES_SUBSET.value] = np.concatenate(
                    [data[key][ModelEnum.GENES_SUBSET.value] for data in batch],
                    axis=0,  # type: ignore
                )
            if ModelEnum.COUNTS_SUBSET.value in batch[0][key]:  # type: ignore
                collated_batch[ModelEnum.COUNTS_SUBSET.value] = np.concatenate(
                    [data[key][ModelEnum.COUNTS_SUBSET.value] for data in batch],
                    axis=0,  # type: ignore
                )
            continue
        if isinstance(batch[0][key], dict):
            subkeys = batch[0][key].keys()  # type: ignore
            if len(batch) > 1 and not all(subkeys == data[key].keys() for data in batch[1:]):  # type: ignore
                raise ValueError(f"All '{key}' sub-dictionaries in the batch must have the same subkeys.")
            # Concatenate all subkeys regardless of their suffix
            value = {
                subkey: np.concatenate([data[key][subkey] for data in batch], axis=0)
                for subkey in subkeys  # type: ignore
            }
        elif key.endswith("_g") or key.endswith("_categories"):
            # Check that all values are the same
            if len(batch) > 1:
                if not all(np.array_equal(batch[0][key], data[key]) for data in batch[1:]):
                    raise ValueError(f"All dictionaries in the batch must have the same {key}.")
            value = batch[0][key]
        else:
            value = np.concatenate([data[key] for data in batch], axis=0)  # type: ignore

        collated_batch[key] = value
    return tree_map(convert_to_tensor, collated_batch)


def tokenize_cells(
    cell: np.ndarray,
    var_names: Sequence[str],
    encoder: VocabularyEncoderSimplified,
    genes_seq_len: int,
    sample_genes: Literal["random", "weighted", "expressed", "expressed_zero", "none"],
    gene_tokens_key: str = ModelEnum.GENES.value,
    counts_key: str = ModelEnum.COUNTS.value,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Tokenize cell counts into gene tokens.

    Parameters
    ----------
    cell
        Count matrix of shape (N, G) where N is number of cells, G is number of genes
    var_names
        Gene names corresponding to columns of cell
    encoder
        Vocabulary encoder to map gene names to token indices
    genes_seq_len
        Maximum sequence length for sampled genes
    sample_genes
        Sampling strategy for genes
    gene_tokens_key
        Key for gene tokens in output dict
    counts_key
        Key for counts in output dict
    seed
        Random seed for reproducibility

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with tokenized genes, counts, and library sizes
    """
    counts = cell
    gene_idx = np.tile(encoder.encode_genes(var_names), (len(counts), 1))
    library_size = counts.sum(1, keepdims=True)

    rng = np.random.default_rng(seed=seed)
    N, G = counts.shape

    if sample_genes == "weighted":
        if encoder.metadata_genes is None:
            raise ValueError("encoder.metadata_genes must be set for weighted sampling")

        scaled_counts = (counts + 1) / encoder.metadata_genes["means"].values
        scaled_counts = scaled_counts / scaled_counts.sum(1, keepdims=True)
        sampled_idx = np.stack([rng.choice(G, size=genes_seq_len, replace=False, p=p) for p in scaled_counts])
        return {
            gene_tokens_key: np.take_along_axis(gene_idx, sampled_idx, axis=1),
            counts_key: np.take_along_axis(counts, sampled_idx, axis=1),
            "library_size": library_size,
        }

    elif sample_genes == "expressed":
        mask_idx = encoder.mask_token_idx
        expressed = counts > 0
        num_expressed = expressed.sum(axis=1)

        if (num_expressed > genes_seq_len).any():
            raise ValueError("genes_seq_len is smaller than number of expressed genes")

        pos_order = expressed.cumsum(axis=1) - 1
        genes_out = np.full((N, genes_seq_len), mask_idx, dtype=gene_idx.dtype)
        counts_out = np.zeros((N, genes_seq_len), dtype=counts.dtype)

        ii, jj = np.where(expressed)
        pp = pos_order[expressed]
        genes_out[ii, pp] = gene_idx[ii, jj]
        counts_out[ii, pp] = counts[ii, jj]

        return {
            gene_tokens_key: gene_idx,
            counts_key: counts,
            ModelEnum.GENES_SUBSET.value: genes_out,
            ModelEnum.COUNTS_SUBSET.value: counts_out,
            "library_size": library_size,
        }

    elif sample_genes == "expressed_zero":
        expressed = counts > 0
        permuted_indices = np.stack([rng.permutation(G) for _ in range(N)])

        shuffled_gene_idx = np.take_along_axis(gene_idx, permuted_indices, axis=1)
        shuffled_counts = np.take_along_axis(counts, permuted_indices, axis=1)
        shuffled_expressed = np.take_along_axis(expressed, permuted_indices, axis=1)

        priority = shuffled_expressed.astype(int)
        sort_indices = np.argsort(priority, axis=1, kind="stable")

        final_gene_idx = np.take_along_axis(shuffled_gene_idx, sort_indices, axis=1)
        final_counts = np.take_along_axis(shuffled_counts, sort_indices, axis=1)

        return {
            gene_tokens_key: gene_idx,
            counts_key: counts,
            ModelEnum.GENES_SUBSET.value: final_gene_idx[:, :genes_seq_len],
            ModelEnum.COUNTS_SUBSET.value: final_counts[:, :genes_seq_len],
            "library_size": library_size,
        }

    elif sample_genes == "random_expressed":
        mask_idx = encoder.mask_token_idx
        nonzero_mask = counts > 0

        sampled_idx = np.stack(
            [
                np.pad(
                    rng.choice(
                        np.nonzero(nonzero_mask[i])[0],
                        size=min(genes_seq_len, nonzero_mask[i].sum()),
                        replace=False,
                    ),
                    (0, max(0, genes_seq_len - nonzero_mask[i].sum())),
                    constant_values=-1,
                )
                for i in range(N)
            ]
        )

        padded_mask = sampled_idx == -1
        safe_sampled_idx = np.where(padded_mask, 0, sampled_idx)

        sampled_gene_idx = np.take_along_axis(gene_idx, safe_sampled_idx, axis=1)
        subset_counts = np.take_along_axis(counts, safe_sampled_idx, axis=1)

        sampled_gene_idx[padded_mask] = mask_idx
        subset_counts[padded_mask] = 0

        return {
            gene_tokens_key: sampled_gene_idx,
            counts_key: subset_counts,
            "library_size": library_size,
        }

    elif sample_genes == "random":
        sampled_idx = np.stack([rng.choice(G, size=genes_seq_len, replace=False) for _ in range(N)])
        return {
            gene_tokens_key: np.take_along_axis(gene_idx, sampled_idx, axis=1),
            counts_key: np.take_along_axis(counts, sampled_idx, axis=1),
            "library_size": library_size,
        }

    elif sample_genes == "none":
        return {
            gene_tokens_key: gene_idx,
            counts_key: counts,
            "library_size": library_size,
        }

    else:
        raise ValueError(f"Invalid sample_genes value: {sample_genes}")


@dataclass
class AnnDataFieldWithVarNames:
    """
    Custom AnnDataField that returns both the data and var_names.

    This is useful when you need both the X data and the corresponding var_names
    for processing, such as for tokenization where you need the actual gene names.
    """

    attr: str
    key: list[str] | str | None = None
    convert_fn: Callable[[dict[str, Any]], np.ndarray] | None = None

    def __call__(self, adata: AnnData) -> np.ndarray:
        from operator import attrgetter

        value = attrgetter(self.attr)(adata)
        if self.key is not None:
            value = value[self.key]

        # Create a dictionary with both the data and var_names
        data_dict = {"data": value.toarray(), "var_names": adata.var_names}

        if self.convert_fn is not None:
            return self.convert_fn(data_dict)
        else:
            return np.asarray(data_dict["data"])


def train_val_split_list(files: list[str], seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.RandomState(seed)
    n_files = len(files)
    n_val_files = int(0.1 * n_files)
    # Only resample from first 50% of files to avoid last file with different cell count
    n_resample = n_files // 2
    indices = np.arange(n_files)
    resample_indices = rng.permutation(n_resample)
    train_indices_arr = np.concatenate([resample_indices[:-n_val_files], indices[n_resample:]])
    val_indices_arr = resample_indices[-n_val_files:]
    return train_indices_arr.tolist(), val_indices_arr.tolist()
