import pytest

import scldm


def test_package_has_version():
    assert scldm.__version__ is not None


def test_imports():
    """Test that main models can be imported."""
    assert hasattr(scldm, "TransformerVAE")
    assert hasattr(scldm, "ScviVAE")
