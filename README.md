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

## Training

### 1. VAE Training

```bash
# Using uv run (automatically manages dependencies)
uv run python experiments/scripts/train.py \
  paths.base_data_path=/path/to/your/data \
  experiment_name=my_vae_experiment \
  training.num_epochs=100

# Or without uv
cd experiments
python scripts/train.py \
  paths.base_data_path=/path/to/your/data \
  experiment_name=my_vae_experiment \
  training.num_epochs=100
```

Key config overrides:
- `paths.base_data_path`: Path to dataset directory
- `experiment_name`: Name for checkpoints/logs
- `datamodule.dataset`: Dataset name (e.g., `dentate_gyrus`)
- `training.num_epochs`: Number of training epochs
- `model.batch_size`: Training batch size

Checkpoints saved to: `experiments/checkpoints/{experiment_name}/`

### 2. Flow Matching (LDM) Training

Requires a trained VAE checkpoint first.

```bash
# Using uv run (automatically manages dependencies)
uv run python experiments/scripts/train_ldm.py \
  paths.base_data_path=/path/to/your/data \
  experiment_name=my_ldm_experiment \
  model.module.vae_as_tokenizer.load_from_checkpoint.ckpt_path=/path/to/vae/checkpoints \
  model.module.vae_as_tokenizer.load_from_checkpoint.job_name=my_vae_experiment

# Or without uv
cd experiments
python scripts/train_ldm.py \
  paths.base_data_path=/path/to/your/data \
  experiment_name=my_ldm_experiment \
  model.module.vae_as_tokenizer.load_from_checkpoint.ckpt_path=/path/to/vae/checkpoints \
  model.module.vae_as_tokenizer.load_from_checkpoint.job_name=my_vae_experiment
```

Key config overrides:
- `model.module.vae_as_tokenizer.load_from_checkpoint.ckpt_path`: Directory containing VAE checkpoint
- `model.module.vae_as_tokenizer.load_from_checkpoint.job_name`: VAE experiment name
- `model.module.vae_as_tokenizer.train`: Set to `true` to fine-tune VAE (default: `false`)

## Inference / Sampling

```bash
# Using uv run (automatically manages dependencies)
uv run python experiments/scripts/inference.py \
  ckpt_file=/path/to/ldm/checkpoint.ckpt \
  config_file=/path/to/ldm/config.yaml \
  datamodule.dataset=dentate_gyrus \
  datamodule.datamodule.test_batch_size=128

# Or without uv
cd experiments
python scripts/inference.py \
  ckpt_file=/path/to/ldm/checkpoint.ckpt \
  config_file=/path/to/ldm/config.yaml \
  datamodule.dataset=dentate_gyrus \
  datamodule.datamodule.test_batch_size=128
```

Key config overrides:
- `ckpt_file`: Path to LDM checkpoint
- `config_file`: Path to saved config.yaml from training
- `model.module.generation_args.guidance_weight`: Classifier-free guidance weight
- `inference_path`: Output directory (default: `outputs/`)

Output: AnnData file saved to `{inference_path}/{dataset}_generated_{idx}.h5ad`

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
