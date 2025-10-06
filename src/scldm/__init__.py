from importlib.metadata import version

from .vae import ScviVAE, TransformerVAE

__all__ = [
    "ScviVAE",
    "TransformerVAE",
]

__version__ = version("scldm")
