import pickle
import types
from pathlib import Path

import hydra
import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import scldm

CHECKPOINTS_PATH = Path("/Users/gpalla/Datasets/scg_vae/vae_census")


class RemapUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("scg_vae"):
            module = module.replace("scg_vae", "scldm")
        return super().find_class(module, name)


remap_pickle = types.ModuleType("remap_pickle")
remap_pickle.Unpickler = RemapUnpickler
remap_pickle.load = pickle.load
remap_pickle.dump = pickle.dump


def load_checkpoint_weights(module, checkpoint_state_dict):
    """Load checkpoint weights into module, handling key mismatches.

    Only loads weights that exist in both checkpoint and module.
    Returns info about loaded/skipped/missing keys.
    """
    module_keys = set(module.state_dict().keys())

    # Filter to only matching keys
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


def test_load_checkpoints():
    config = OmegaConf.load(CHECKPOINTS_PATH / "20M.yaml")

    # Replace scg_vae with scldm in config (in-memory only, disk untouched)
    config_str = OmegaConf.to_yaml(config)
    config_str = config_str.replace("scg_vae", "scldm")
    config = OmegaConf.create(config_str)

    # Instantiate module
    module = hydra.utils.instantiate(config.model.module)
    print(f"Module type: {type(module).__name__}")

    # Load checkpoint with remapping (in-memory only, disk untouched)
    with open(CHECKPOINTS_PATH / "20M.ckpt", "rb") as f:
        checkpoint = torch.load(f, map_location="cpu", pickle_module=remap_pickle)

    # Load weights
    stats = load_checkpoint_weights(module, checkpoint["state_dict"])

    print(f"✓ Checkpoint: {stats['loaded']} loaded, {stats['skipped']} skipped, {stats['missing']} missing")
