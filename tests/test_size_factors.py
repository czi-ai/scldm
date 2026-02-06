import pickle
from pathlib import Path

import numpy as np
import pytest

from scldm._utils import compute_log_size_factor_stats


@pytest.mark.requires_local_data
def test_dentate_gyrus_size_factors_match_artifacts() -> None:
    repo_root = Path(__file__).parent.parent
    train_path = repo_root / "_artifacts" / "datasets" / "dentategyrus_train.h5ad"
    mu_path = repo_root / "_artifacts" / "resubmission" / "dentate_gyrus_log_size_factor_mu.pkl"
    sd_path = repo_root / "_artifacts" / "resubmission" / "dentate_gyrus_log_size_factor_sd.pkl"

    if not (train_path.exists() and mu_path.exists() and sd_path.exists()):
        pytest.skip(
            "Local dentategyrus artifacts not found. "
            "Download with: python -m scldm.download_artifacts --group datasets --group resubmission"
        )

    mu_stats, sd_stats = compute_log_size_factor_stats(train_path, ["clusters"])

    with mu_path.open("rb") as f:
        expected_mu = pickle.load(f)
    with sd_path.open("rb") as f:
        expected_sd = pickle.load(f)

    assert set(mu_stats.keys()) == {"clusters"}
    assert set(sd_stats.keys()) == {"clusters"}
    assert mu_stats["clusters"].keys() == expected_mu["clusters"].keys()
    assert sd_stats["clusters"].keys() == expected_sd["clusters"].keys()

    mu_values = np.array([mu_stats["clusters"][k] for k in expected_mu["clusters"].keys()], dtype=float)
    mu_expected = np.array([expected_mu["clusters"][k] for k in expected_mu["clusters"].keys()], dtype=float)
    sd_values = np.array([sd_stats["clusters"][k] for k in expected_sd["clusters"].keys()], dtype=float)
    sd_expected = np.array([expected_sd["clusters"][k] for k in expected_sd["clusters"].keys()], dtype=float)

    np.testing.assert_allclose(mu_values, mu_expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sd_values, sd_expected, rtol=1e-6, atol=1e-6)
