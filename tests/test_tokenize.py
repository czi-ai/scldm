"""Tests for tokenize_cells function."""

import numpy as np
import pytest

from scldm.constants import ModelEnum
from scldm.datamodule import tokenize_cells
from scldm.encoder import VocabularyEncoderSimplified


@pytest.fixture
def mock_encoder():
    """Create a mock encoder for testing."""

    class MockEncoder:
        def __init__(self):
            self.mask_token_idx = 0
            self.metadata_genes = None

        def encode_genes(self, var_names):
            return np.arange(1, len(var_names) + 1)

    return MockEncoder()


@pytest.fixture
def sample_data():
    """Create sample count data for testing."""
    counts = np.array(
        [
            [5, 0, 3, 0, 2],
            [0, 4, 0, 1, 0],
            [2, 1, 0, 0, 3],
        ]
    )
    var_names = ["gene_a", "gene_b", "gene_c", "gene_d", "gene_e"]
    return counts, var_names


def test_tokenize_none(mock_encoder, sample_data):
    """Test tokenization with no sampling."""
    counts, var_names = sample_data
    result = tokenize_cells(counts, var_names, mock_encoder, genes_seq_len=5, sample_genes="none")

    assert ModelEnum.GENES.value in result
    assert ModelEnum.COUNTS.value in result
    assert "library_size" in result

    assert result[ModelEnum.GENES.value].shape == (3, 5)
    assert result[ModelEnum.COUNTS.value].shape == (3, 5)
    assert result["library_size"].shape == (3, 1)

    np.testing.assert_array_equal(result[ModelEnum.COUNTS.value], counts)
    np.testing.assert_array_equal(result["library_size"], counts.sum(1, keepdims=True))


def test_tokenize_random(mock_encoder, sample_data):
    """Test random sampling."""
    counts, var_names = sample_data
    result = tokenize_cells(counts, var_names, mock_encoder, genes_seq_len=3, sample_genes="random", seed=42)

    assert result[ModelEnum.GENES.value].shape == (3, 3)
    assert result[ModelEnum.COUNTS.value].shape == (3, 3)
    assert result["library_size"].shape == (3, 1)

    for i in range(3):
        assert result[ModelEnum.COUNTS.value][i].sum() <= counts[i].sum()


def test_tokenize_expressed(mock_encoder, sample_data):
    """Test expressed genes sampling."""
    counts, var_names = sample_data
    result = tokenize_cells(counts, var_names, mock_encoder, genes_seq_len=5, sample_genes="expressed")

    assert ModelEnum.GENES_SUBSET.value in result
    assert ModelEnum.COUNTS_SUBSET.value in result

    genes_subset = result[ModelEnum.GENES_SUBSET.value]
    counts_subset = result[ModelEnum.COUNTS_SUBSET.value]

    assert genes_subset.shape == (3, 5)
    assert counts_subset.shape == (3, 5)

    for i in range(3):
        nonzero_counts = counts_subset[i][counts_subset[i] > 0]
        original_nonzero = counts[i][counts[i] > 0]
        assert len(nonzero_counts) == len(original_nonzero)


def test_tokenize_expressed_zero(mock_encoder, sample_data):
    """Test sampling from unexpressed genes first."""
    counts, var_names = sample_data
    result = tokenize_cells(
        counts,
        var_names,
        mock_encoder,
        genes_seq_len=3,
        sample_genes="expressed_zero",
        seed=42,
    )

    assert ModelEnum.GENES_SUBSET.value in result
    assert ModelEnum.COUNTS_SUBSET.value in result

    genes_subset = result[ModelEnum.GENES_SUBSET.value]
    counts_subset = result[ModelEnum.COUNTS_SUBSET.value]

    assert genes_subset.shape == (3, 3)
    assert counts_subset.shape == (3, 3)


def test_tokenize_random_expressed(mock_encoder, sample_data):
    """Test random sampling from expressed genes only."""
    counts, var_names = sample_data
    result = tokenize_cells(
        counts,
        var_names,
        mock_encoder,
        genes_seq_len=3,
        sample_genes="random_expressed",
        seed=42,
    )

    assert result[ModelEnum.GENES.value].shape == (3, 3)
    assert result[ModelEnum.COUNTS.value].shape == (3, 3)

    for i in range(3):
        sampled_counts = result[ModelEnum.COUNTS.value][i]
        original_expressed = counts[i] > 0
        num_expressed = original_expressed.sum()

        if num_expressed >= 3:
            assert (sampled_counts > 0).sum() == 3
        else:
            assert (sampled_counts > 0).sum() == num_expressed


def test_tokenize_weighted_no_metadata(mock_encoder, sample_data):
    """Test weighted sampling fails without metadata."""
    counts, var_names = sample_data
    with pytest.raises(ValueError, match="encoder.metadata_genes must be set"):
        tokenize_cells(
            counts,
            var_names,
            mock_encoder,
            genes_seq_len=3,
            sample_genes="weighted",
        )


def test_tokenize_expressed_too_many(mock_encoder, sample_data):
    """Test expressed sampling fails when too many expressed genes."""
    counts, var_names = sample_data
    with pytest.raises(ValueError, match="genes_seq_len is smaller"):
        tokenize_cells(
            counts,
            var_names,
            mock_encoder,
            genes_seq_len=2,
            sample_genes="expressed",
        )


def test_tokenize_invalid_method(mock_encoder, sample_data):
    """Test invalid sampling method raises error."""
    counts, var_names = sample_data
    with pytest.raises(ValueError, match="Invalid sample_genes value"):
        tokenize_cells(
            counts,
            var_names,
            mock_encoder,
            genes_seq_len=3,
            sample_genes="invalid",
        )


def test_tokenize_library_size_correct(mock_encoder, sample_data):
    """Test library size is correctly computed."""
    counts, var_names = sample_data
    result = tokenize_cells(counts, var_names, mock_encoder, genes_seq_len=5, sample_genes="none")

    expected_lib_size = counts.sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(result["library_size"], expected_lib_size)


def test_tokenize_batch_shape(mock_encoder):
    """Test with different batch sizes."""
    for n_cells in [1, 5, 10]:
        counts = np.random.randint(0, 10, size=(n_cells, 100))
        var_names = [f"gene_{i}" for i in range(100)]

        result = tokenize_cells(counts, var_names, mock_encoder, genes_seq_len=50, sample_genes="random", seed=42)

        assert result[ModelEnum.GENES.value].shape == (n_cells, 50)
        assert result[ModelEnum.COUNTS.value].shape == (n_cells, 50)
        assert result["library_size"].shape == (n_cells, 1)
