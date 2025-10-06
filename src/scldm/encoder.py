import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd


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
        if self.adata_path is not None:
            self.adata = ad.read_h5ad(self.adata_path)
        else:
            self.adata = None
        if self.metadata_genes is not None:
            self.metadata_genes = pd.read_parquet(self.metadata_genes)
            self.genes = self.metadata_genes["feature_id"].values
            # Create conversion dict from gene symbols to ensemble genes
            self.gene_symbol_to_ensembl = dict(
                zip(self.metadata_genes["feature_name"].values, self.metadata_genes["feature_id"].values, strict=False)
            )
        else:
            self.genes = self.adata.var_names.values

        # Auto-detect n_genes if not provided or mismatched
        detected_n_genes = len(self.genes)
        if self.n_genes is None:
            self.n_genes = detected_n_genes
        elif self.n_genes != detected_n_genes:
            # Prefer the adata var_names length to avoid runtime failures
            self.n_genes = detected_n_genes

        if self.adata is not None:
            self.labels = {
                label: self.adata.obs[label].cat.categories.tolist() for label in self.class_vocab_sizes.keys()
            }
        else:
            self.labels = None

        genes_tokens = ["<MASK>"]
        genes_tokens += list(self.genes)

        self._gene_token2idx = {token: idx for idx, token in enumerate(map(str, genes_tokens))}
        self._gene_idx2token = dict(enumerate(genes_tokens))

        self.gene_tokens_idx = list(self._gene_token2idx.values())[1:]
        assert self.mask_token_idx == self._gene_token2idx[self.mask_token]

        if self.labels is not None:
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
