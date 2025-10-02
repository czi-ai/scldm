import os
import pathlib
from collections.abc import Callable

import hydra
from omegaconf import DictConfig, OmegaConf

os.environ["HYDRA_FULL_ERROR"] = "1"


def train(cfg) -> None:
    import pytorch_lightning as pl
    import torch
    import torch.distributed as dist
    from hydra.core.hydra_config import HydraConfig
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DDPStrategy
    from scg_vae._train_utils import (
        PreallocatedAccumulator,
        compute_model_stats,
        setup_callbacks_and_loggers_and_paths,
        setup_datamodule_and_steps,
        setup_wandb_run,
    )
    from scg_vae._utils import world_info_from_env
    from scg_vae.logger import logger

    torch.set_float32_matmul_precision("high")

    if hasattr(cfg, "dataset_generation_idx"):
        if isinstance(cfg.dataset_generation_idx, str):
            pl.seed_everything(cfg.seed + len(cfg.dataset_generation_idx))
        else:
            pl.seed_everything(cfg.seed + cfg.dataset_generation_idx)
    else:
        pl.seed_everything(cfg.seed)
    logger.info("Set distributed environment...")

    local_rank, global_rank, world_size = world_info_from_env()
    logger.info(f"LOCAL RANK {local_rank}, GLOBAL RANK {global_rank}, WORLD SIZE {world_size}")

    # TODO: maybe remove this eventually
    torch._dynamo.config.capture_scalar_outputs = True

    # get env variables from runai
    NUM_NODES = int(os.environ.get("PET_NNODES", 1))
    NUM_GPU_PER_DEVICE = int(os.environ.get("RUNAI_NUM_OF_GPUS", 1))
    WORLD_SIZE_ENV = int(os.environ.get("WORLD_SIZE") or "1")  # Fix: Handle None case properly
    WORLD_SIZE = NUM_NODES * NUM_GPU_PER_DEVICE
    if WORLD_SIZE_ENV is None:
        WORLD_SIZE_ENV = WORLD_SIZE
    print(f"WORLD_SIZE: {WORLD_SIZE}, WORLD_SIZE_ENV: {WORLD_SIZE_ENV}, world_size: {world_size}")
    # assert WORLD_SIZE == WORLD_SIZE_ENV == world_size, "WORLD SIZE MISMATCH"

    # Initialize the distributed environment
    # Set the device before initializing process group
    torch.cuda.set_device(local_rank)
    if WORLD_SIZE > 1:
        dist.init_process_group(backend="nccl")
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
    is_main_process = (world_size == 1) or (global_rank == 0)

    logger.info("Loading original configs...")
    original_cfg = OmegaConf.load(os.path.join(cfg.training.callbacks.model_checkpoints.dirpath, "config.yaml"))

    # # specific rewrite for fm inference
    if hasattr(original_cfg.model.module, "vae_as_tokenizer"):
        original_cfg.model.module.vae_as_tokenizer = None
    #     original_cfg.model.module.norm_factor = cfg.model.module.norm_factor
    #     logger.info(f"norm_factor: {original_cfg.model.module.norm_factor}")
    original_cfg.model.batch_size = cfg.model.batch_size

    if hasattr(cfg.model.module, "inference_args"):
        original_cfg.model.module.inference_args = cfg.model.module.inference_args
    if hasattr(cfg.model.module, "generation_args"):
        original_cfg.model.module.generation_args = cfg.model.module.generation_args

    test_batch_size = cfg.model.test_batch_size
    batch_size = cfg.model.batch_size
    cfg.model = original_cfg.model
    cfg.model.test_batch_size = test_batch_size
    cfg.model.batch_size = batch_size
    # Merge original datamodule with overrides so we can change dataset/paths via this run's config
    cfg.datamodule = OmegaConf.merge(original_cfg.datamodule, cfg.datamodule)

    logger.info("Instantiating datamodule...")
    print(f"DATAMODULE CONFIG: {cfg.datamodule}")
    datamodule = setup_datamodule_and_steps(cfg, WORLD_SIZE, cfg.training.num_epochs)
    logger.info(f"Effective number of steps {cfg.training.trainer.max_steps}")
    datamodule.setup()
    print(f"DATAMODULE TYPE: {type(datamodule).__name__}")

    logger.info("Instantiating module...")

    module = hydra.utils.instantiate(cfg.model.module)

    module.vae_model.eval()

    if hasattr(cfg, "temperature"):
        module.vae_model.decoder_head.t = cfg.temperature
        logger.info(f"Temperature set to: {module.vae_model.decoder_head.t}")

    # Compute normalizer factor for latent diffusion
    # norm_factor_path = os.path.join(cfg.training.callbacks.model_checkpoints.dirpath, "normalize_factor.pt")
    # if is_main_process and original_cfg.model.module.norm_factor is None:
    #     logger.info("Computing normalizer factor for VAE latents...")
    #     normalizer = module.compute_normalizer(
    #         dataloader=datamodule.train_dataloader(),
    #         norm_factor_path=norm_factor_path,
    #     )
    # else:
    #     normalizer = 0.0  # placeholder for non-main processes

    # # Synchronize normalizer across all processes
    # if dist.is_initialized():
    #     normalizer_tensor = torch.tensor(normalizer, device=torch.cuda.current_device())
    #     dist.broadcast(normalizer_tensor, src=0)
    #     normalizer = normalizer_tensor.item()
    #     dist.barrier()

    # Assign the synchronized normalizer to the VAE model
    # module.vae_model.norm_factor = 0.66968032905521
    # logger.info(f"Normalizer factor set to: {module.vae_model.norm_factor}")

    # first, get run name
    # get overrides if it exists
    if len(HydraConfig.get().job.override_dirname):
        import re

        # Split by commas that are not inside square brackets
        override_items = re.split(r",(?![^\[]*\])", HydraConfig.get().job.override_dirname)
        override_dict = dict(item.split("=", 1) for item in override_items)
        override_dict = {k.replace("+", ""): v for k, v in override_dict.items()}
    else:
        override_dict = {"foo": "bar"}  # if no overrides used, e.g. for testing

    logger.info("Instantiating callbacks and logger...")
    callbacks_, loggers_, cfg = setup_callbacks_and_loggers_and_paths(cfg, override_dict, inference=True)
    # Default: preallocate arrays and write final H5AD at end of predict
    output_filename = getattr(getattr(cfg, "inference", {}), "output_filename", "adata_test_generated.h5ad")
    output_path = os.path.join(cfg.training.callbacks.model_checkpoints.dirpath, output_filename)
    callbacks_["predict_preallocated_accumulator"] = PreallocatedAccumulator(
        output_path=output_path,
        datamodule=datamodule,
        is_main_process=is_main_process,
    )

    cfg = compute_model_stats(cfg, module)
    logger.info(f"Model flops: {cfg.model.flops} parameters: {cfg.model.num_parameters}")
    override_dict["num_parameters"] = cfg.model.num_parameters
    override_dict["flops"] = cfg.model.flops

    # Save the config to the checkpoint directory
    config_path = os.path.join(cfg.training.callbacks.model_checkpoints.dirpath, "config.yaml")
    # with open(config_path, "w") as f:
    #     OmegaConf.save(cfg, f)
    # logger.info(f"Configuration saved to {config_path}")

    if cfg.training.logger.wandb is not None:
        run_id_to_use = setup_wandb_run(is_main_process=is_main_process, cfg=cfg, global_rank=global_rank)
        group = override_dict.get("seed", "default")
        loggers_["wandb"] = loggers_["wandb"](
            id=run_id_to_use,
            group=group,
        )
        if is_main_process:
            loggers_["wandb"].experiment.config.update(override_dict, allow_val_change=True)
            loggers_["wandb"].experiment.save(config_path)

    trainer_: Callable[..., Trainer] = hydra.utils.instantiate(cfg.training.trainer)  # partial

    if "strategy" not in cfg.training.trainer:
        strategy: str | DDPStrategy = "ddp" if WORLD_SIZE > 1 else "auto"
        if strategy == "ddp":
            strategy = DDPStrategy(
                gradient_as_bucket_view=True,
                static_graph=False,
                find_unused_parameters=False,
            )
    else:
        strategy = cfg.training.trainer.strategy

    trainer: Trainer = trainer_(
        num_nodes=NUM_NODES,
        devices=1,  # NUM_GPU_PER_DEVICE,
        strategy=strategy,
        use_distributed_sampler=False,
        logger=list(loggers_.values()),
        callbacks=list(callbacks_.values()),
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
        if dist.is_initialized():
            dist.barrier()
        if is_main_process:
            torch.cuda.empty_cache()
            torch._dynamo.config.cache_size_limit = 1000
            if cfg.model.module.generation_args is not None or cfg.model.module.inference_args is not None:
                # # PREDICT
                # Load checkpoint into the existing module
                # TODO check with GIO
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

                module.load_state_dict(checkpoint["state_dict"])
                module.eval()

                datamodule.setup("predict")
                output = trainer.predict(
                    module,
                    # datamodule.predict_dataloader(),
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
                ckpt_path=ckpt_path,
                return_predictions=False,
            )
            logger.info(f"Predict finished. Preallocated outputs saved to {output_path}")
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

        try:
            rank_to_report = dist.get_rank() if dist.is_initialized() else 0
        except RuntimeError:
            rank_to_report = 0
        logger.error(f"An error occurred on rank {rank_to_report}")
        logger.error(traceback.format_exc())
        raise


@hydra.main(
    config_path="./../config/",
    config_name="parse_1m_fm_joint_gc_1_keep_out.yaml",
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

# torchrun --nnodes 1 --nproc-per-node 1 experiments/scripts/inference.py
