import hydra
import pytest
import torch
from omegaconf import OmegaConf

from scldm._utils import remap_pickle


def _load_checkpoint_weights(module, checkpoint_state_dict):
    """Load checkpoint weights into module, handling key mismatches.

    Only loads weights that exist in both checkpoint and module.
    Returns info about loaded/skipped/missing keys.

    Parameters
    ----------
    module
        PyTorch module to load weights into
    checkpoint_state_dict
        State dict from checkpoint file

    Returns
    -------
    dict
        Dictionary with counts of loaded, skipped, and missing keys
    """
    module_keys = set(module.state_dict().keys())
    filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in module_keys}

    loaded_keys = set(filtered_state_dict.keys())
    skipped_keys = set(checkpoint_state_dict.keys()) - loaded_keys
    missing_keys = module_keys - loaded_keys

    if skipped_keys:
        print(f"Skipping {len(skipped_keys)} keys not in module: {list(skipped_keys)[:5]}...")
    if missing_keys:
        print(f"Module has {len(missing_keys)} keys not in checkpoint: {list(missing_keys)[:5]}...")
    print(f"Loading {len(loaded_keys)} matching keys")

    module.load_state_dict(filtered_state_dict, strict=False)

    return {"loaded": len(loaded_keys), "skipped": len(skipped_keys), "missing": len(missing_keys)}


@pytest.mark.requires_local_data
def test_load_checkpoints(checkpoints_path):
    """Test loading pretrained checkpoint with scg_vae → scldm remapping.

    Parameters
    ----------
    checkpoints_path
        Path to checkpoint directory containing 20M.yaml and 20M.ckpt
    """
    if not checkpoints_path.exists():
        pytest.skip(f"Checkpoint path not found: {checkpoints_path}")

    config_file = checkpoints_path / "20M.yaml"
    checkpoint_file = checkpoints_path / "20M.ckpt"

    if not config_file.exists() or not checkpoint_file.exists():
        pytest.skip("Required checkpoint files not found")

    # Load config and remap scg_vae → scldm (in-memory only)
    config = OmegaConf.load(config_file)
    config_str = OmegaConf.to_yaml(config)
    config_str = config_str.replace("scg_vae", "scldm")
    config = OmegaConf.create(config_str)

    # Instantiate module
    module = hydra.utils.instantiate(config.model.module)
    print(f"Module type: {type(module).__name__}")

    # Load checkpoint with remapping (in-memory only)
    with open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu", pickle_module=remap_pickle)

    # Load weights
    stats = _load_checkpoint_weights(module, checkpoint["state_dict"])

    print(f"✓ Checkpoint: {stats['loaded']} loaded, {stats['skipped']} skipped, {stats['missing']} missing")


@pytest.mark.requires_local_data
def test_datamodule_load(dentategyrus_paths):
    """Test DataModule initialization with dentategyrus dataset using Hydra config.

    Parameters
    ----------
    dentategyrus_paths
        Tuple of (train_path, test_path) for dentategyrus dataset
    """
    from pathlib import Path

    train_path, test_path = dentategyrus_paths

    if not train_path.exists() or not test_path.exists():
        pytest.skip(f"Dentategyrus dataset not found at {train_path.parent}")

    # Load configs
    config_dir = Path(__file__).parent.parent / "experiments" / "configs"
    paths_config = OmegaConf.load(config_dir / "paths" / "default.yaml")
    datamodule_config = OmegaConf.load(config_dir / "datamodule" / "default.yaml")

    # Merge configs with proper structure
    config = OmegaConf.create({"paths": paths_config, "datamodule": datamodule_config})

    # Instantiate vocabulary encoder and datamodule
    vocab_encoder = hydra.utils.instantiate(config.datamodule.vocabulary_encoder)
    datamodule = hydra.utils.instantiate(config.datamodule.datamodule)

    # Setup should handle train/val split
    datamodule.setup(stage="fit")

    assert hasattr(datamodule, "train_dataset"), "Missing train_dataset after setup"
    assert hasattr(datamodule, "val_dataset"), "Missing val_dataset after setup"

    print(f"✓ DataModule created successfully from config")
    print(f"  Dataset: {config.datamodule.dataset}")
    print(f"  Train cells: {datamodule.n_cells}")
    print(f"  Has train dataset: {hasattr(datamodule, 'train_dataset')}")
    print(f"  Has val dataset: {hasattr(datamodule, 'val_dataset')}")
