from pathlib import Path

import anndata as ad
import numpy as np
import pytest

# Local data paths (only available on specific machine)
CHECKPOINTS_PATH = Path("/Users/gpalla/Datasets/scg_vae/vae_census")
DENTATEGYRUS_TRAIN_PATH = Path("/Users/gpalla/Datasets/scg_vae/dentategyrus/dentategyrus_train.h5ad")
DENTATEGYRUS_TEST_PATH = Path("/Users/gpalla/Datasets/scg_vae/dentategyrus/dentategyrus_test.h5ad")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_local_data: tests that require local dataset files")


@pytest.fixture
def checkpoints_path():
    """Path to pretrained model checkpoints."""
    return CHECKPOINTS_PATH


@pytest.fixture
def dentategyrus_paths():
    """Paths to dentategyrus train/test datasets."""
    return DENTATEGYRUS_TRAIN_PATH, DENTATEGYRUS_TEST_PATH
