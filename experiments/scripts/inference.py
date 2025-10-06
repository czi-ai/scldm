import os
import pathlib
from collections.abc import Callable

import hydra
from omegaconf import DictConfig, OmegaConf

from scldm._utils import (
    process_generation_output,
    process_inference_output,
    remap_config,
    remap_pickle,
)

os.environ["HYDRA_FULL_ERROR"] = "1"


def train(cfg) -> None:
    import pytorch_lightning as pl
    import torch
    from hydra.core.hydra_config import HydraConfig
    from pytorch_lightning import Trainer

    from scldm.logger import logger

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.capture_scalar_outputs = True

    pl.seed_everything(cfg.seed)
    logger.info("Single-process inference mode")

    logger.info("Loading module config from disk...")
    module_ckpt = pathlib.Path(cfg.ckpt_file)
    module_config = OmegaConf.load(pathlib.Path(cfg.config_file))

    logger.info("Instantiating datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule.datamodule)

    logger.info("Instantiating module from module config...")
    remap_config(module_config)
    module = hydra.utils.instantiate(module_config.model.module)

    module.vae_model.eval()

    if len(HydraConfig.get().job.override_dirname):
        import re

        # Split by commas that are not inside square brackets
        override_items = re.split(r",(?![^\[]*\])", HydraConfig.get().job.override_dirname)
        override_dict = dict(item.split("=", 1) for item in override_items)
        override_dict = {k.replace("+", ""): v for k, v in override_dict.items()}
    else:
        override_dict = {"foo": "bar"}  # if no overrides used, e.g. for testing

    logger.info("Instantiating callbacks and loggers from config...")
    callbacks_list = []
    for cb_cfg in getattr(cfg.training, "callbacks", {}).values():
        # Force full instantiation (avoid functools.partial)
        callbacks_list.append(hydra.utils.instantiate(cb_cfg, _partial_=False))

    loggers_list = []
    for lg_cfg in getattr(cfg.training, "logger", {}).values():
        # Force full instantiation (avoid functools.partial)
        loggers_list.append(hydra.utils.instantiate(lg_cfg, _partial_=False))

    trainer_: Callable[..., Trainer] = hydra.utils.instantiate(cfg.training.trainer)  # partial

    trainer: Trainer = trainer_(
        devices=1,
        strategy="auto",
        logger=loggers_list if len(loggers_list) > 0 else False,
        callbacks=callbacks_list,
    )

    if module_ckpt.exists():
        logger.info(f"Loading checkpoint weights from {module_ckpt}")
        checkpoint = torch.load(module_ckpt, map_location="cpu", pickle_module=remap_pickle, weights_only=False)
        state_dict = (
            checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        )
        module_keys = set(module.state_dict().keys())

        # Load only matching keys into the module
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in module_keys}
        loaded = len(filtered_state_dict)
        skipped = len(state_dict) - loaded
        missing = len(module_keys) - loaded

        if skipped:
            logger.info(f"Skipping {skipped} keys not present in module")
        if missing:
            logger.info(f"Module has {missing} keys not in checkpoint")
        logger.info(f"Loading {loaded} matching keys from checkpoint")

        module.load_state_dict(filtered_state_dict, strict=False)
    else:
        logger.warning(f"Checkpoint file not found: {module_ckpt}")

    torch.cuda.empty_cache()
    torch._dynamo.config.cache_size_limit = 1000
    if cfg.model.module.generation_args is not None or cfg.model.module.inference_args is not None:
        # # PREDICT
        module.eval()
        datamodule.setup("predict")
        output = trainer.predict(
            module,
            datamodule=datamodule,
        )

        if cfg.model.module.generation_args is not None:
            adata = process_generation_output(output, datamodule)
        else:
            adata = process_inference_output(output, datamodule)

        inference_type = "generated" if cfg.model.module.generation_args is not None else "inference"
        adata.obs["generation_idx"] = cfg.dataset_generation_idx
        save_path = (
            pathlib.Path(cfg.inference_path)
            / f"{cfg.datamodule.dataset}_{inference_type}_{cfg.dataset_generation_idx}.h5ad"
        )
        adata.write(save_path)


@hydra.main(
    config_path="./../configs/",
    config_name="inference.yaml",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass
    train(cfg)


if __name__ == "__main__":
    main()

# Run single-process inference:
# python experiments/scripts/inference.py
