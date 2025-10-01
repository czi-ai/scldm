from enum import StrEnum


class DatasetEnum(StrEnum):
    """Enum for dataset keys."""

    TISSUE = "tissue"
    TISSUE_GENERAL = "tissue_general"
    DONOR_ID = "donor_id"
    ASSAY = "assay"  #
    SUSPENSION_TYPE = "suspension_type"  #
    DATASET_ID = "dataset_id"
    NNZ = "nnz"
    RAW_SUM = "raw_sum"  #
    N_MEASURED_VARS = "n_measured_vars"
    SEX = "sex"
    DISEASE = "disease"
    DEVELOPMENT_STAGE = "development_stage"
    CELL_TYPE = "cell_type"
    SUSPENSION = "suspension_type"


class ModelEnum(StrEnum):
    """Enum for model keys."""

    COUNTS = "counts"  #
    GENES = "genes"  #
    LIBRARY_SIZE = "library_size"
    GENES_SUBSET = "genes_subset"
    COUNTS_SUBSET = "counts_subset"


class LossEnum(StrEnum):
    """Enum for model keys."""

    LLH_LOSS = "llh"
    KL_LOSS = "kl"
    DIFF_LOSS = "diff"
    CR_LOSS = "cr"
