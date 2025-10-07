# scLDM: Single-Cell Latent Diffusion Model

## Overview

scLDM is a deep learning framework for modeling single-cell gene expression using variational autoencoders (VAEs) and latent diffusion models. The package provides state-of-the-art architectures for learning compressed representations of single-cell transcriptomics data.

## Key Components

### Core VAE Architectures

- **{class}`~scldm.TransformerVAE`**: Transformer-based VAE architecture for modeling gene expression patterns

### Training Modules

- **{class}`~scldm.models.VAE`**: PyTorch Lightning module for training VAE models
- **{class}`~scldm.models.LatentDiffusion`**: Latent diffusion model for generative modeling in latent space

### Data Handling

- **{class}`~scldm.datamodule.DataModule`**: PyTorch Lightning DataModule for loading and preprocessing single-cell datasets

## Quick Links

- {doc}`API Reference <api>` - Detailed documentation of classes and functions
- {doc}`Example Notebook <notebooks/example>` - Tutorial on using scLDM
- {doc}`Contributing <contributing>` - Guidelines for contributing to the project

---

```{include} ../README.md

```

```{toctree}
:hidden: true
:maxdepth: 1

api.md
changelog.md
contributing.md
references.md

notebooks/example
```
