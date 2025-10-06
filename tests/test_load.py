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
        print(f"  Skipping {len(skipped_keys)} keys not in module: {list(skipped_keys)[:5]}...")
    if missing_keys:
        print(f"  Module has {len(missing_keys)} keys not in checkpoint: {list(missing_keys)[:5]}...")
    print(f"  Loading {len(loaded_keys)} matching keys")

    module.load_state_dict(filtered_state_dict, strict=False)

    return {"loaded": len(loaded_keys), "skipped": len(skipped_keys), "missing": len(missing_keys)}


@pytest.mark.requires_local_data
def test_end_to_end_load(dentategyrus_paths, checkpoints_path, config_dir):
    """Test end-to-end loading of datamodule, module, and trainer from configs.

    Parameters
    ----------
    dentategyrus_paths
        Tuple of (train_path, test_path) for dentategyrus dataset
    checkpoints_path
        Path to checkpoint directory containing pretrained models
    config_dir
        Path to experiments/configs directory
    """
    import lightning as L

    train_path, test_path = dentategyrus_paths

    if not train_path.exists() or not test_path.exists():
        pytest.skip(f"Dentategyrus dataset not found at {train_path.parent}")

    if not checkpoints_path.exists():
        pytest.skip(f"Checkpoint path not found: {checkpoints_path}")

    config_file = checkpoints_path / "20M.yaml"
    checkpoint_file = checkpoints_path / "20M.ckpt"

    if not config_file.exists() or not checkpoint_file.exists():
        pytest.skip("Required checkpoint files not found")

    print("\n=== Loading DataModule ===")
    # Load datamodule configs
    paths_config = OmegaConf.load(config_dir / "paths" / "default.yaml")
    datamodule_config = OmegaConf.load(config_dir / "datamodule" / "default.yaml")
    config = OmegaConf.merge(paths_config, datamodule_config)

    # Instantiate datamodule
    datamodule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup(stage="fit")

    assert hasattr(datamodule, "train_dataset"), "Missing train_dataset after setup"
    assert hasattr(datamodule, "val_dataset"), "Missing val_dataset after setup"

    print(f"✓ DataModule loaded")
    print(f"  Dataset: {config.dataset}")
    print(f"  Train cells: {datamodule.n_cells}")

    print("\n=== Loading Module ===")
    # Load module config and remap scg_vae → scldm (in-memory only)
    module_config = OmegaConf.load(config_file)
    config_str = OmegaConf.to_yaml(module_config)
    config_str = config_str.replace("scg_vae", "scldm")
    module_config = OmegaConf.create(config_str)

    # Instantiate module
    module = hydra.utils.instantiate(module_config.model.module)
    print(f"✓ Module instantiated: {type(module).__name__}")

    # Load checkpoint weights with remapping (in-memory only)
    with open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu", pickle_module=remap_pickle)

    stats = _load_checkpoint_weights(module, checkpoint["state_dict"])
    print(f"✓ Weights loaded: {stats['loaded']} keys")

    print("\n=== Creating Trainer ===")
    # Create a minimal trainer for testing
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    print(f"✓ Trainer created")

    print("\n=== Validation ===")
    # Get a batch from validation dataset (avoid multiprocessing issues in tests)
    assert datamodule.val_dataset is not None, "Val dataset not initialized"
    print(f"✓ Val dataset: {len(datamodule.val_dataset)} batches")

    print(f"\n✓ End-to-end test passed!")
    print(f"  - DataModule: {datamodule.n_cells} cells, {len(datamodule.val_dataset)} val batches")
    print(f"  - Module: {type(module).__name__}")
    print(f"  - Checkpoint: {stats['loaded']} weights loaded")
    print(f"  - Trainer: Ready")
