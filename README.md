# scldm

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/czi-ai/scldm/test.yaml?branch=main
[badge-docs]: https://img.shields.io/github/actions/workflow/status/czi-ai/scldm/docs.yaml?branch=main

single-cell latent diffusion model

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

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/czi-ai/scldm/issues
[tests]: https://github.com/czi-ai/scldm/actions/workflows/test.yaml
[documentation]: https://czi-ai.github.io/scldm
[changelog]: https://czi-ai.github.io/scldm/changelog.html
[api documentation]: https://czi-ai.github.io/scldm/api.html
[pypi]: https://pypi.org/project/scldm
