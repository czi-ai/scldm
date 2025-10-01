import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd

from scldm.constants import DatasetEnum


@dataclass
class VocabularyEncoderSimplified:
    """Encode a vocabulary of genes and labels into indices."""

    adata_path: ad.AnnData
    class_vocab_sizes: dict[str, int]
    mask_token: str = "<MASK>"
    mask_token_idx: int = 0
    n_genes: int | None = None
    guidance_weight: dict[str, float] | None = None
    mu_size_factor: Path | str | None = None
    sd_size_factor: Path | str | None = None
    condition_strategy: Literal["mutually_exclusive", "joint"] = "mutually_exclusive"
    metadata_genes: Path | str | None = None

    _token2idx: dict[str, int] = field(init=False, repr=False)
    _idx2token: dict[int, str] = field(init=False, repr=False)

    def __post_init__(self):
        self.adata = ad.read_h5ad(self.adata_path)
        if self.metadata_genes is not None:
            self.metadata_genes = pd.read_parquet(self.metadata_genes)
            self.genes = self.metadata_genes["feature_id"].values
        else:
            self.genes = self.adata.var_names.values

        # Auto-detect n_genes if not provided or mismatched
        detected_n_genes = len(self.genes)
        if self.n_genes is None:
            self.n_genes = detected_n_genes
        elif self.n_genes != detected_n_genes:
            # Prefer the adata var_names length to avoid runtime failures
            self.n_genes = detected_n_genes

        self.labels = {label: self.adata.obs[label].cat.categories.tolist() for label in self.class_vocab_sizes.keys()}

        genes_tokens = ["<MASK>"]
        genes_tokens += list(self.genes)

        self._gene_token2idx = {token: idx for idx, token in enumerate(map(str, genes_tokens))}
        self._gene_idx2token = dict(enumerate(genes_tokens))

        self.gene_tokens_idx = list(self._gene_token2idx.values())[1:]
        assert self.mask_token_idx == self._gene_token2idx[self.mask_token]

        self.classes2idx = {
            label: {token: idx for idx, token in enumerate(map(str, self.labels[label]))}
            for label in self.class_vocab_sizes.keys()
        }
        self.idx2classes = {
            label: {idx: token for token, idx in self.classes2idx[label].items()}
            for label in self.class_vocab_sizes.keys()
        }

        # size factors
        if hasattr(self, "condition_strategy") and self.condition_strategy != "joint":
            if self.mu_size_factor is not None:
                mu_size_factor_dict = pickle.load(open(self.mu_size_factor, "rb"))
                self.mu_size_factor = {}
                for label in self.class_vocab_sizes.keys():
                    self.mu_size_factor[label] = {
                        self.classes2idx[label][k]: v for k, v in mu_size_factor_dict[label].items()
                    }

            if self.sd_size_factor is not None:
                sd_size_factor_dict = pickle.load(open(self.sd_size_factor, "rb"))
                self.sd_size_factor = {}
                for label in self.class_vocab_sizes.keys():
                    self.sd_size_factor[label] = {
                        self.classes2idx[label][k]: v for k, v in sd_size_factor_dict[label].items()
                    }
        elif hasattr(self, "condition_strategy") and self.condition_strategy == "joint":
            if self.mu_size_factor is not None:
                mu_size_factor_dict = pickle.load(open(self.mu_size_factor, "rb"))
                self.mu_size_factor = {}
                self.mu_size_factor["cell_type_cytokine"] = mu_size_factor_dict["cell_type_cytokine"]
                self.joint_idx_2_classes = {}
                for _idx, token in enumerate(mu_size_factor_dict["cell_type_cytokine"].keys()):
                    # get cell_type and cytokine from token
                    cell_type, cytokine = token.split("_")

                    cell_type_idx = self.classes2idx["cell_line"][cell_type]
                    cytokine_idx = self.classes2idx["gene"][cytokine]
                    self.joint_idx_2_classes[str(cell_type_idx) + "_" + str(cytokine_idx)] = token

            if self.sd_size_factor is not None:
                sd_size_factor_dict = pickle.load(open(self.sd_size_factor, "rb"))
                self.sd_size_factor = {}
                self.sd_size_factor["cell_type_cytokine"] = sd_size_factor_dict["cell_type_cytokine"]

            # handle idx mapping later during generation time
        # Remove adata reference as it's no longer needed after initialization
        del self.adata
        self.adata = None

    def encode_genes(self, tokens: Sequence[str]) -> np.ndarray:
        """Convert tokens to their corresponding indices.

        Ensures a numeric dtype output. Unknown tokens map to the mask token index.
        """
        mask_idx = self.mask_token_idx
        indices = [self._gene_token2idx.get(str(token), mask_idx) for token in tokens]
        return np.asarray(indices, dtype=np.int64)

    def decode_genes(self, indices: Sequence[int]) -> np.ndarray:
        """Convert indices back to their corresponding tokens."""
        return np.vectorize(lambda idx: self._gene_idx2token.get(idx, None))(indices)

    def encode_metadata(self, metadata: Sequence[str], label: str) -> np.ndarray:
        return np.array([self.classes2idx[label].get(str(item), None) for item in metadata])

    def decode_metadata(self, indices: Sequence[int], label: str) -> np.ndarray:
        return np.array([self.idx2classes[label].get(item, None) for item in indices])


@dataclass
class VocabularyEncoder:
    """
    Encode a vocabulary of genes and labels into indices.

    The indices are assigned in the following order:
    1. The padding token
    2. The special tokens
    3. The classes
    4. The genes

    Parameters
    ----------
    adata:
        The path to an anndata object. If provided, the genes and labels will be extracted from the anndata object.
    genes:
        The genes to encode.
    labels:
        The labels to encode.
    pad_token:
        The token to use as padding.
    special_tokens:
        The special tokens to use.
    """

    adata: Path | str | ad.AnnData | None = None
    metadata_cells: Path | str | pd.DataFrame | None = None
    metadata_genes: Path | str | pd.DataFrame | None = None
    labels: Sequence[str] | Mapping[str, list[str]] | None = None
    genes: Sequence[str] | None = None
    n_genes: int | None = None
    mask_token: str = "<MASK>"
    mask_token_idx: int = 0
    assay_vocab_size: int | None = None
    suspension_vocab_size: int | None = None

    # Internal state stored after initialization
    _token2idx: dict[str, int] = field(init=False, repr=False)
    _idx2token: dict[int, str] = field(init=False, repr=False)

    def __post_init__(self):
        if self.adata is not None:
            # read anndata to get classes
            if isinstance(self.adata, Path) or isinstance(self.adata, str):
                self.adata = ad.read_h5ad(self.adata)
            assert isinstance(self.adata, ad.AnnData), (
                "either pass a valid path to an anndata object or an anndata object"
            )

        if isinstance(self.metadata_cells, Path) or isinstance(self.metadata_cells, str):
            self.metadata_cells = pd.read_parquet(self.metadata_cells)
        assert isinstance(self.metadata_cells, pd.DataFrame), (
            "either pass a valid path to a metadata file or a pandas dataframe"
        )

        if isinstance(self.metadata_genes, Path) or isinstance(self.metadata_genes, str):
            self.metadata_genes = pd.read_parquet(self.metadata_genes)
        assert isinstance(self.metadata_genes, pd.DataFrame), (
            "either pass a valid path to a metadata file or a pandas dataframe"
        )

        if self.labels is not None:
            self.labels = {label: self.metadata_cells[label].unique().tolist() for label in self.labels}

        # self.genes = self.adata.var_names.values
        self.genes = self.metadata_genes["feature_id"].values
        if len(self.genes) != self.n_genes:
            raise ValueError(f"Number of genes in adata ({len(self.genes)}) does not match n_genes ({self.n_genes})")
        np.testing.assert_array_equal(self.genes, self.metadata_genes["feature_id"].values)

        if self.genes is None:
            raise ValueError("Genes must be provided, either via adata or genes.")

        if self.labels is not None:
            self.classes2idx = {
                label: {token: idx for idx, token in enumerate(map(str, self.labels[label]))}
                for label in self.labels.keys()
            }
            self.idx2classes = {
                label: {idx: token for token, idx in self.classes2idx[label].items()} for label in self.labels.keys()
            }

        assert len(self.labels[DatasetEnum.ASSAY.value]) == self.assay_vocab_size
        assert len(self.labels[DatasetEnum.SUSPENSION.value]) == self.suspension_vocab_size

        genes_tokens = ["<MASK>"]
        genes_tokens += list(self.genes)

        self._gene_token2idx = {token: idx for idx, token in enumerate(map(str, genes_tokens))}
        self._gene_idx2token = dict(enumerate(genes_tokens))

        self.gene_tokens_idx = list(self._gene_token2idx.values())[1:]
        assert self.mask_token_idx == self._gene_token2idx[self.mask_token]

    def encode_metadata(self, metadata: Sequence[str], label: str) -> np.ndarray:
        return np.array([self.classes2idx[label].get(item, None) for item in metadata])

    def decode_metadata(self, indices: Sequence[int], label: str) -> np.ndarray:
        return np.array([self.idx2classes[label].get(idx) if idx is not None else None for idx in indices])

    def encode_genes(self, tokens: Sequence[str], batch: bool = False) -> np.ndarray:
        """Convert tokens to their corresponding indices. If batch is True, encode batch-wise."""
        if batch:
            return np.array([np.vectorize(lambda token: self._gene_token2idx.get(token))(seq) for seq in tokens])
        else:
            return np.vectorize(lambda token: self._gene_token2idx.get(token))(tokens)

    def decode_genes(self, indices: Sequence[int], batch: bool = False) -> np.ndarray:
        """Convert indices back to their corresponding tokens. If batch is True, decode batch-wise."""
        if batch:
            return np.array([np.vectorize(lambda idx: self._gene_idx2token.get(idx))(seq) for seq in indices])
        else:
            return np.vectorize(lambda idx: self._gene_idx2token.get(idx))(indices)


IS_ASSAY_UMI_BASED = {
    "10x 3' transcription profiling": True,
    "10x 3' v1": True,
    "10x 3' v2": True,
    "10x 3' v3": True,
    "10x 5' transcription profiling": True,
    "10x 5' v1": True,
    "10x 5' v2": True,
    "BD Rhapsody Targeted mRNA": True,
    "BD Rhapsody Whole Transcriptome Analysis": True,
    "CEL-seq2": True,
    "Drop-seq": True,
    "GEXSCOPE technology": False,
    "MARS-seq": True,
    "SPLiT-seq": True,
    "STRT-seq": True,
    "ScaleBio single cell RNA sequencing": True,
    "Seq-Well": True,
    "Seq-Well S3": True,
    "Smart-seq v4": False,
    "Smart-seq2": False,
    "TruDrop": True,  # https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-020-06843-0
    "inDrop": True,
    "microwell-seq": True,
    "sci-RNA-seq": True,
}
