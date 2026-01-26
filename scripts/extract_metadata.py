import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd
import yaml


def normalize_dataset_name(name: str) -> str:
    return name.replace("_", "").replace("-", "").lower()


def resolve_dataset_key(file_stem: str, dataset_params: dict) -> str:
    base = file_stem
    for suffix in ("_train", "_test"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    normalized = normalize_dataset_name(base)
    for key in dataset_params.keys():
        if normalize_dataset_name(key) == normalized:
            return key
    raise ValueError(f"Could not match dataset for file stem '{file_stem}' in config dataset_params")


def load_dataset_params(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dataset_params = config.get("dataset_params")
    if not isinstance(dataset_params, dict):
        raise ValueError("Config missing dataset_params mapping")
    return dataset_params


def ensure_categorical(series: pd.Series, label: str) -> pd.Categorical:
    if not isinstance(series.dtype, pd.CategoricalDtype):
        raise ValueError(f"obs column '{label}' must be categorical to preserve ordering")
    return series.cat


def extract_metadata_for_file(adata_path: Path, dataset_key: str, label_keys: list[str]) -> dict:
    adata = ad.read_h5ad(adata_path)
    genes = adata.var_names.tolist()

    labels = {}
    for label in label_keys:
        if label not in adata.obs:
            raise ValueError(f"obs column '{label}' not found in {adata_path}")
        categories = ensure_categorical(adata.obs[label], label).categories.tolist()
        labels[label] = categories

    return {
        "genes": genes,
        "labels": labels,
        "dataset": dataset_key,
        "source_h5ad": str(adata_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract gene/class metadata from AnnData files into JSON.")
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("_artifacts/datasets"),
        help="Directory containing .h5ad datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scldm/metadata"),
        help="Output directory for JSON metadata files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scldm/experiments/configs/datamodule/default.yaml"),
        help="Path to config with dataset_params and class_vocab_sizes",
    )
    args = parser.parse_args()

    dataset_params = load_dataset_params(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    h5ad_files = sorted(args.datasets_dir.glob("*.h5ad"))
    if not h5ad_files:
        raise ValueError(f"No .h5ad files found in {args.datasets_dir}")

    for adata_path in h5ad_files:
        dataset_key = resolve_dataset_key(adata_path.stem, dataset_params)
        class_vocab_sizes = dataset_params[dataset_key].get("class_vocab_sizes")
        label_keys = list(class_vocab_sizes.keys()) if isinstance(class_vocab_sizes, dict) else []

        metadata = extract_metadata_for_file(adata_path, dataset_key, label_keys)
        output_path = args.output_dir / f"{adata_path.stem}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
