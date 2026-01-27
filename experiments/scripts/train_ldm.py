import os
import pathlib

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from scldm._utils import load_validate_statedict_config, setup_datamodule_and_steps
from scldm.logger import logger

os.environ["HYDRA_FULL_ERROR"] = "1"


def train(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed)

    # Get world size from environment (set by torchrun/lightning)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    # Setup datamodule and compute training steps
    datamodule = setup_datamodule_and_steps(cfg, world_size, cfg.training.num_epochs)
    datamodule.setup()

    # Scale learning rate by world size (for diffusion optimizer, not VAE)
    if world_size > 1:
        if "diffusion_optimizer" in cfg.model.module:
            original_lr = cfg.model.module.diffusion_optimizer.lr
            cfg.model.module.diffusion_optimizer.lr = original_lr * world_size
            logger.info(f"Scaled diffusion LR: {original_lr} -> {cfg.model.module.diffusion_optimizer.lr}")

    # Check if using VAE as tokenizer (loading from checkpoint)
    is_vae_as_tokenizer = (
        hasattr(cfg.model.module, "vae_as_tokenizer") and "load_from_checkpoint" in cfg.model.module.vae_as_tokenizer
    )

    vae_state_dict = None
    if is_vae_as_tokenizer:
        ckpt_cfg = cfg.model.module.vae_as_tokenizer.load_from_checkpoint
        job_path = pathlib.Path(f"{ckpt_cfg.ckpt_path}/{ckpt_cfg.job_name}")

        # Determine checkpoint file
        checkpoint_file = (
            f"epoch={ckpt_cfg.epoch}.ckpt"
            if ckpt_cfg.epoch is not None and isinstance(ckpt_cfg.epoch, int)
            else "last.ckpt"
        )

        logger.info(f"Loading VAE checkpoint from: {job_path / checkpoint_file}")

        # Load checkpoint and config
        vae_checkpoints = torch.load(job_path / checkpoint_file, weights_only=False)
        vae_config = OmegaConf.load(job_path / "config.yaml")

        # Extract VAE state dict and update config
        vae_state_dict, cfg = load_validate_statedict_config(vae_checkpoints, cfg, vae_config)
        logger.info(f"Loaded VAE config from checkpoint, train mode: {cfg.model.module.vae_as_tokenizer.train}")

    # Instantiate model
    logger.info("Instantiating model...")
    module = hydra.utils.instantiate(cfg.model.module)

    # Load VAE weights if using as tokenizer
    if is_vae_as_tokenizer and vae_state_dict is not None:
        module.vae_model.load_state_dict(vae_state_dict)
        logger.info("VAE model weights loaded from checkpoint")

        # Freeze VAE if not training
        if not cfg.model.module.vae_as_tokenizer.train:
            for param in module.vae_model.parameters():
                param.requires_grad = False
            module.vae_model.eval()
            logger.info("VAE model frozen (train=false)")

    # Setup callbacks
    callbacks = []
    for cb_name, cb_cfg in cfg.training.callbacks.items():
        callbacks.append(hydra.utils.instantiate(cb_cfg))
        logger.info(f"Added callback: {cb_name}")

    # Setup loggers
    loggers = []
    for lg_name, lg_cfg in cfg.training.logger.items():
        if lg_name == "wandb":
            if local_rank == 0:
                wandb_partial = hydra.utils.instantiate(lg_cfg)
                loggers.append(wandb_partial(id=None))
                logger.info(f"Added logger: {lg_name}")
        else:
            loggers.append(hydra.utils.instantiate(lg_cfg))
            logger.info(f"Added logger: {lg_name}")

    # Create trainer
    trainer_partial = hydra.utils.instantiate(cfg.training.trainer)
    strategy = DDPStrategy(find_unused_parameters=False) if world_size > 1 else "auto"

    trainer: Trainer = trainer_partial(
        devices="auto",
        strategy=strategy,
        logger=loggers if loggers else False,
        callbacks=callbacks,
        use_distributed_sampler=False,
    )

    # Save config
    checkpoint_dir = pathlib.Path(cfg.training.callbacks.model_checkpoints.dirpath)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if local_rank == 0:
        OmegaConf.save(cfg, checkpoint_dir / "config.yaml")
        logger.info(f"Saved config to {checkpoint_dir / 'config.yaml'}")

    # Resume from checkpoint if exists
    last_ckpt = checkpoint_dir / "last.ckpt"
    ckpt_path = str(last_ckpt) if last_ckpt.exists() else None
    if ckpt_path:
        logger.info(f"Resuming from checkpoint: {ckpt_path}")

    # Train
    logger.info("Starting LDM training...")
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info("Training complete!")


@hydra.main(config_path="../configs", config_name="ldm_training", version_base="1.2")
def main(cfg: DictConfig) -> None:
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass
    train(cfg)


if __name__ == "__main__":
    main()
