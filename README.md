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

### Environment setup (required before `uv run`)

`uv run` assumes you already have an environment with dependencies installed. Because
`cellarium-ml` must be installed from source, set up the environment first:

```bash
uv venv
source .venv/bin/activate
uv pip install -e . "cellarium-ml @ git+https://github.com/cellarium-ai/cellarium-ml.git"
```

If you prefer, you can use `pip` in an existing environment instead of `uv pip`.

## Checkpoints and other artifacts

To download model checkpoints and other required artifacts:

```bash
scldm-download-artifacts --group resubmission
# or (after environment setup)
uv run scldm-download-artifacts --group resubmission
```

Downloads come from the public `s3://czi-scldm` bucket using unsigned requests by
default. Files are placed under the `scldm/_artifacts` directory unless you override
`--destination`.

We recommend downloading only `--group resubmission`, since it includes the primary
checkpoints and configs. The `vae_census` artifacts are separate and unchanged, so only
download `--group vae_census` if you need those. The `datasets` group contains the
`dentategyrus` train/test AnnData files. You can pass `--group` multiple times or use
comma-separated values (use `all` to fetch everything).

## Dataset files (AnnData) and metadata

Train/test AnnData paths are defined in `experiments/configs/paths/datasets.yaml` and are
expected to live under `paths.base_data_path` (override this path as needed). The default
relative layout is:

- `dentate_gyrus`: `dentategyrus_train.h5ad`, `dentategyrus_test.h5ad`
- `hlca`: `hlca_train_sharded/adata_0.h5ad`, `hlca_test_sharded/adata_0.h5ad`
- `tabula_muris`: `tabula_muris_train_sharded/adata_0.h5ad`, `tabula_muris_test_sharded/adata_0.h5ad`
- `parse1m`: `parse1m_train.h5ad`, `parse1m_test.h5ad`
- `replogle`: `replogle_train.h5ad`, `replogle_test.h5ad`

Download sources for these files are referenced in the inline comments of
`experiments/configs/paths/datasets.yaml` (CFGen figshare for `hlca`/`tabula_muris`,
Parse1m figshare, and Replogle figshare/GEO). These datasets are not fetched by
`scldm-download-artifacts`, so you must download them separately and place them at the paths
above (or override `paths.base_data_path` and dataset paths in Hydra).

The JSON files under `metadata/` are required to define perturbation splits for the
`parse1m` and `replogle` datasets, so make sure they are present when running those configs.

## Training

### 1. VAE Training

```bash
# Using uv run (after environment setup above)
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
# Using uv run (after environment setup above)
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
# Using uv run (after environment setup above)
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

## Pretrained class embeddings (DiT)

To load pretrained conditional class embeddings for the diffusion model, provide a `.pt`
file containing a `state_dict` and `labels` mapping. The `labels` list must be ordered
by the embedding index and will be strictly validated against the label encoder.

Minimal payload format:

```python
payload = {
    "state_dict": {
        "class_embeddings.cell_type.weight": weight,  # shape: [num_classes + cfg, n_embed]
    },
    "labels": {
        "cell_type": ["B", "T"],  # ordered by index
    },
}
torch.save(payload, "class_embeds.pt")
```

Example Hydra override:

```bash
model.module.diffusion_model.pretrained_class_embeddings.ckpt_path=/path/to/class_embeds.pt \
model.module.diffusion_model.pretrained_class_embeddings.freeze=true
```

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
