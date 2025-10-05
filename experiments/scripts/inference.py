import os
import pathlib
from collections.abc import Callable

import hydra
from omegaconf import DictConfig, OmegaConf

os.environ["HYDRA_FULL_ERROR"] = "1"


def train(cfg) -> None:
    import pytorch_lightning as pl
    import torch
    from hydra.core.hydra_config import HydraConfig
    from pytorch_lightning import Trainer

    from scldm.logger import logger

    torch.set_float32_matmul_precision("high")

    pl.seed_everything(cfg.seed)
    logger.info("Single-process inference mode")

    logger.info("Loading module config from disk...")
    module_config = pathlib.Path(cfg.training.callbacks.model_checkpoints.dirpath) / "config.yaml"
    module_cfg_from_disk = OmegaConf.load(module_config)

    logger.info("Instantiating datamodule...")
    print(f"DATAMODULE CONFIG: {cfg.datamodule}")
    # Instantiate datamodule directly from runtime config
    try:
        datamodule = hydra.utils.instantiate(cfg.datamodule.datamodule)
    except Exception:
        datamodule = hydra.utils.instantiate(cfg.datamodule)
    logger.info(f"Effective number of steps {cfg.training.trainer.max_steps}")
    datamodule.setup()
    print(f"DATAMODULE TYPE: {type(datamodule).__name__}")

    logger.info("Instantiating module from module config...")
    # Build model strictly from disk-loaded module config, while preserving runtime overrides for args
    if hasattr(module_cfg_from_disk.model.module, "vae_as_tokenizer"):
        module_cfg_from_disk.model.module.vae_as_tokenizer = None
    if hasattr(cfg.model.module, "inference_args"):
        module_cfg_from_disk.model.module.inference_args = cfg.model.module.inference_args
    if hasattr(cfg.model.module, "generation_args"):
        module_cfg_from_disk.model.module.generation_args = cfg.model.module.generation_args
    module = hydra.utils.instantiate(module_cfg_from_disk.model.module)

    module.vae_model.eval()

    if hasattr(cfg, "temperature"):
        module.vae_model.decoder_head.t = cfg.temperature
        logger.info(f"Temperature set to: {module.vae_model.decoder_head.t}")

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
    try:
        for cb_cfg in getattr(cfg.training, "callbacks", {}).values():
            try:
                callbacks_list.append(hydra.utils.instantiate(cb_cfg))
            except Exception:
                pass
    except Exception:
        pass
    loggers_list = []
    try:
        for lg_cfg in getattr(cfg.training, "logger", {}).values():
            try:
                loggers_list.append(hydra.utils.instantiate(lg_cfg))
            except Exception:
                pass
    except Exception:
        pass

    # Save the config to the checkpoint directory
    config_path = os.path.join(cfg.training.callbacks.model_checkpoints.dirpath, "config.yaml")
    # with open(config_path, "w") as f:
    #     OmegaConf.save(cfg, f)
    # logger.info(f"Configuration saved to {config_path}")

    # Optional: wandb logger will be configured via Hydra if present

    trainer_: Callable[..., Trainer] = hydra.utils.instantiate(cfg.training.trainer)  # partial

    trainer: Trainer = trainer_(
        devices=1,
        strategy="auto",
        logger=loggers_list if len(loggers_list) > 0 else False,
        callbacks=callbacks_list,
    )
    try:
        checkpoint_dir = pathlib.Path(cfg.training.callbacks.model_checkpoints.dirpath)
        last_checkpoint = checkpoint_dir / cfg.ckpt_file
        logger.info(f"Loading checkpoint from {last_checkpoint}")
        print(f"EXACT FILE BEING LOADED: {last_checkpoint.absolute()}")
        print(f"FILE EXISTS: {last_checkpoint.exists()}")
        ckpt_path = last_checkpoint if last_checkpoint.exists() else None
        if ckpt_path is not None:
            print(f"CONFIRMED: Loading checkpoint file: {ckpt_path.absolute()}")
            # maybe_fix_compiled_weights(ckpt_path, diffusion_compile=cfg.model.module.compile, vae_compile=False)
            # Load checkpoint and filter state dict to only include keys present in the module
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            checkpoint_state_dict = checkpoint["state_dict"]
            module_keys = set(module.state_dict().keys())

            # Filter checkpoint state dict to only include keys that exist in the module
            filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in module_keys}

            # Log which keys are being loaded vs skipped
            loaded_keys = set(filtered_state_dict.keys())
            skipped_keys = set(checkpoint_state_dict.keys()) - loaded_keys
            missing_keys = module_keys - loaded_keys

            if skipped_keys:
                logger.info(f"Skipping {len(skipped_keys)} keys not present in module: {list(skipped_keys)[:5]}...")
            if missing_keys:
                logger.info(f"Module has {len(missing_keys)} keys not in checkpoint: {list(missing_keys)[:5]}...")
            logger.info(f"Loading {len(loaded_keys)} matching keys from checkpoint")

            # Load the filtered state dict
            module.load_state_dict(filtered_state_dict, strict=False)
            ckpt_path = None  # Set to None so trainer doesn't try to load again
        else:
            print(f"WARNING: Checkpoint file not found at {last_checkpoint.absolute()}")
            print(
                f"Checkpoint directory contents: {list(checkpoint_dir.iterdir()) if checkpoint_dir.exists() else 'Directory does not exist'}"
            )
        # trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
        if is_main_process:
            torch.cuda.empty_cache()
            torch._dynamo.config.cache_size_limit = 1000
            if cfg.model.module.generation_args is not None or cfg.model.module.inference_args is not None:
                # # PREDICT
                module.eval()
                datamodule.setup("predict")
                _ = trainer.predict(
                    module,
                    datamodule=datamodule,
                )

                # Filter out None values from output (from our single batch limitation)

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
            print(f"cfg.datamodule has 'adata_inference_path': {hasattr(cfg.datamodule, 'adata_inference_path')}")

            inference_path = None
            if hasattr(cfg.datamodule, "datamodule") and hasattr(cfg.datamodule.datamodule, "adata_inference_path"):
                inference_path = cfg.datamodule.datamodule.adata_inference_path
                print(f"FOUND INFERENCE PATH IN NESTED CONFIG: {inference_path}")
            elif hasattr(cfg.datamodule, "adata_inference_path"):
                inference_path = cfg.datamodule.adata_inference_path
                print(f"FOUND INFERENCE PATH IN TOP-LEVEL CONFIG: {inference_path}")

            if inference_path is not None:
                import anndata as ad

                logger.info(f"Loading adata_inference from {inference_path}")
                print(f"EXACT DATA FILE BEING LOADED: {inference_path}")
                datamodule.adata_inference = ad.read_h5ad(inference_path)
                print(f"DATA FILE LOADED SUCCESSFULLY: {inference_path}")
                print(f"DATA SHAPE: {datamodule.adata_inference.shape}")
            else:
                print("NO CUSTOM DATA FILE SPECIFIED - using default datamodule data")
            print("SETTING UP DATAMODULE FOR PREDICTION...")
            datamodule.setup("predict")
            print("PREDICTION DATAMODULE SETUP COMPLETE")
            _ = trainer.predict(
                module,
                # datamodule.predict_dataloader(),
                datamodule=datamodule,
                return_predictions=False,
            )
            logger.info("Predict finished.")
            # # TEST
            # datamodule.setup("test")
            # trainer = Trainer(
            #     devices=1,
            #     logger=list(loggers_.values()),
            #     callbacks=list(callbacks_.values()),
            # )
            # test_results = trainer.test(module, dataloaders=datamodule.test_dataloader())
            # logger.info(test_results)

    except Exception:
        import traceback

        rank_to_report = 0
        logger.error(f"An error occurred on rank {rank_to_report}")
        logger.error(traceback.format_exc())
        raise


@hydra.main(
    config_path="./../configs/",
    config_name="dentate_gyrus_inference.yaml",
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
