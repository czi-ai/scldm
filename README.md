# scldm

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/czi-ai/scldm/test.yaml?branch=main
[badge-docs]: https://img.shields.io/github/actions/workflow/status/czi-ai/scldm/docs.yaml?branch=main

`scLDM` is a latent-diffusion model consisting of a novel fully transformer-based VAE architecture for exchangeable data that uses a single set of fixed-size, permutation-invariant latent variables. The model introduces a Multi-head Cross-Attention Block (MCAB) that serves dual purposes: It acts as a permutation-invariant pooling operator in the encoder, and functions as a permutation-equivariant unpooling operator in the decoder. This unified approach eliminates the need for separate architectural components for handling varying set sizes. Our latent diffusion model is trained with the flow matching loss and linear interpolants using the Scalable Interpolant Transformers formulation (SiT) (Ma et al., 2024), and a denoiser parameterized by Diffusion Transformers (DiT) (Peebles & Xie, 2023). This allows for better modeling of the complex distribution of cellular states and enables controlled generation through classifier-free guidance.

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

### Note On  Dependencies

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

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse](https://discourse.scverse.org/).
If you found a bug, please use the [issue tracker]().

## Citation

> Palla G., Babu S., Dibaeinia P., Li D., Khan A., Karaletsos T., Tomczak J.M., Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models, arXiv, 2025

[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/czi-ai/scldm/issues
[tests]: https://github.com/czi-ai/scldm/actions/workflows/test.yaml
[documentation]: https://czi-ai.github.io/scldm
[changelog]: https://czi-ai.github.io/scldm/changelog.html
[api documentation]: https://czi-ai.github.io/scldm/api.html
[pypi]: https://pypi.org/project/scldm
