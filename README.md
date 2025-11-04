# scldm

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]
[![pre-commit.ci status][badge-precommit]][pre-commit]
[![Release][badge-release]][release]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/czi-ai/scldm/test.yaml?branch=main
[badge-docs]: https://img.shields.io/github/actions/workflow/status/czi-ai/scldm/docs.yaml?branch=main
[badge-precommit]: https://results.pre-commit.ci/badge/github/czi-ai/scldm/main.svg
[badge-release]: https://github.com/czi-ai/scldm/actions/workflows/release.yaml/badge.svg?event=release

`scLDM` is a latent-diffusion model consisting of a novel fully transformer-based VAE architecture for exchangeable data that uses a single set of fixed-size, permutation-invariant latent variables. The model introduces a Multi-head Cross-Attention Block (MCAB) that serves dual purposes: It acts as a permutation-invariant pooling operator in the encoder, and functions as a permutation-equivariant unpooling operator in the decoder. This unified approach eliminates the need for separate architectural components for handling varying set sizes. Our latent diffusion model is trained with the flow matching loss and linear interpolants using the Scalable Interpolant Transformers formulation (SiT) (Ma et al., 2024), and a denoiser parameterized by Diffusion Transformers (DiT) (Peebles & Xie, 2023). This allows for better modeling of the complex distribution of cellular states and enables controlled generation through classifier-free guidance.

![scldm-banner](https://github.com/czi-ai/scldm/blob/main/docs/_static/main.png?raw=true)

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.11 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

To install the latest release of `scldm` from [PyPI][]:

```bash
pip install scldm "cellarium-ml @ git+https://github.com/cellarium-ai/cellarium-ml.git"
# or
uv pip install scldm "cellarium-ml @ git+https://github.com/cellarium-ai/cellarium-ml.git"
```


### Note On Dependencies

#### Cellarium-ML

This model uses [cellarium-ml](https://github.com/cellarium-ai/cellarium-ml). Currently,
the most recent version on PyPI (0.0.7) is not compatible with `anndata>=0.10.9`,
which this model uses. You must install a newer version of `cellarium-ml` from source:

You can install cellarium-ml separately with:

```bash
pip install "cellarium-ml @ git+https://github.com/cellarium-ai/cellarium-ml.git"
# or
uv pip install "cellarium-ml @ git+https://github.com/cellarium-ai/cellarium-ml.git"
```

## Checkpoints and other artifacts

To download model checkpoints and other required artifacts:

```bash
scldm-download-artifacts
# or
uv run scldm-download-artifacts
```

This will automatically download all artifacts to the `_artifacts` subdirectory. You
can change this with the `--destination` flag. If you don't want to download all
files, you can specify `--group datasets`, `--group vae_census`, `--group fm_observational`, and/or
`--group fm_perturbation` to download just those artifacts.


## Release notes

See the [changelog][].

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> Palla G., Babu S., Dibaeinia P., Li D., Khan A., Karaletsos T., Tomczak J.M., Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models, arXiv, 2025

[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/czi-ai/scldm/issues
[tests]: https://github.com/czi-ai/scldm/actions/workflows/test.yaml
[pre-commit]: https://results.pre-commit.ci/latest/github/czi-ai/scldm/main
[documentation]: https://czi-ai.github.io/scldm
[release]: https://github.com/scverse/spatialdata/actions/workflows/release.yaml
[changelog]: https://czi-ai.github.io/scldm/changelog.html
[api documentation]: https://czi-ai.github.io/scldm/api.html
[pypi]: https://pypi.org/project/scldm
