from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from ema_pytorch import EMA
from hydra.core.config_store import DictConfig
from pytorch_lightning import LightningModule
from scvi.distributions import NegativeBinomial as NegativeBinomialSCVI
from torch.distributions import Distribution, Normal
from torch.utils._pytree import tree_map
from torchmetrics.functional.regression import mean_squared_error, pearson_corrcoef, r2_score

from scldm.constants import LossEnum, ModelEnum
from scldm.distributions import log_nb_positive
from scldm.evaluations import (
    BrayCurtisKernel,
    MMDLoss,
    RBFKernel,
    RuzickaKernel,
    TanimotoKernel,
    wasserstein,
)
from scldm.logger import logger
from scldm.nnets import DiT
from scldm.transport import Sampler, Transport
from scldm.vae import TransformerVAE

REGRESSION_METRICS = {
    "mse": mean_squared_error,
    "pcc": pearson_corrcoef,
    # "scc": spearman_corrcoef,
    # "r2": partial(r2_score, multioutput="raw_values"),
}

MMD_METRICS = {
    "mmd_braycurtis_counts": MMDLoss(kernel=BrayCurtisKernel()),
    "mmd_tanimoto": MMDLoss(kernel=TanimotoKernel()),
    "mmd_ruzicka_counts": MMDLoss(kernel=RuzickaKernel()),
    "mmd_rbf": MMDLoss(kernel=RBFKernel()),
}

WASSERSTEIN_METRICS = {
    "wasserstein1_sinkhorn": partial(wasserstein, method="sinkhorn", power=1),
    "wasserstein2_sinkhorn": partial(wasserstein, method="sinkhorn", power=2),
}


R2_METRICS = {
    "r2_mean": lambda preds, target: r2_score(preds.mean(0), target.mean(0)),
    "r2_var": lambda preds, target: r2_score(preds.var(0), target.var(0)),
}


class BaseModel(LightningModule, ABC):
    """Abstract base class for VAE-based models."""

    @abstractmethod
    def sample(self, *args, **kwargs) -> torch.Tensor:
        """Sample from the model."""
        pass

    @abstractmethod
    def inference(self, *args, **kwargs) -> dict[str, Any]:
        """Inference from the model."""
        pass

    def validation_step(self, batch: dict[str, torch.Tensor | dict[str, torch.Tensor]], batch_idx: int) -> None:
        metrics = self.shared_step(batch, batch_idx, "val")
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        metrics = self.shared_step(batch, batch_idx, "val", ema=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch: dict[str, torch.Tensor | dict[str, torch.Tensor]], batch_idx: int) -> None:
        metrics = self.shared_step(batch, batch_idx, "test")
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        metrics = self.shared_step(batch, batch_idx, "test", ema=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Now the parameters have been updated by the optimizer
        # This is the right place to update the EMA
        if hasattr(self, "ema_model"):
            self.ema_model.update()

    def on_train_epoch_start(self) -> None:
        # from cellarium-ml
        combined_loader = self.trainer.fit_loop._combined_loader
        assert combined_loader is not None
        dataloaders = combined_loader.flattened
        for dataloader in dataloaders:
            dataset = dataloader.dataset
            set_epoch = getattr(dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(self.current_epoch)

    def on_validation_start(self):
        """Reset datasets before validation to ensure consistent state"""
        # Add logging to debug
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            logger.info(f"Rank {rank}/{world_size} - Validation starting")
            # train_dataloader = self.trainer.train_dataloader()
            val_dataloader = (
                self.trainer.val_dataloaders[0]
                if isinstance(self.trainer.val_dataloaders, list)
                else self.trainer.val_dataloaders
            )
            if val_dataloader is not None:
                logger.info(f"Rank {rank}/{world_size} - Val dataset size: {len(val_dataloader.dataset)}")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        fit_loop = self.trainer.fit_loop
        epoch_loop = fit_loop.epoch_loop
        batch_progress = epoch_loop.batch_progress
        if batch_progress.current.completed < batch_progress.current.processed:  # type: ignore[attr-defined]
            # Checkpointing is done before these attributes are updated. So, we need to update them manually.
            checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"]["completed"] += 1
            checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"]["completed"] += 1
            if not epoch_loop._should_accumulate():
                checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"]["_batches_that_stepped"] += 1
            if batch_progress.is_last_batch:
                checkpoint["loops"]["fit_loop"]["epoch_progress"]["total"]["processed"] += 1
                checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]["processed"] += 1
                checkpoint["loops"]["fit_loop"]["epoch_progress"]["total"]["completed"] += 1
                checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] += 1

    def _compute_gradient_norms(self, modules: dict[str, nn.Module]) -> dict[str, float]:
        """Compute gradient norms for each module."""
        grad_norms = {}

        # Compute norms for each module and their submodules
        for name, module in modules.items():
            if module is None or not any(p.requires_grad for p in module.parameters()):
                continue

            # Total norm for the module
            grad_norms[f"grad_norm/{name}"] = self._calculate_grad_norm(module.parameters())

            # Compute norms for each submodule
            for submodule_name, submodule in module.named_children():
                # Compute norms for each sub-submodule
                for sub_submodule_name, sub_submodule in submodule.named_children():
                    if not any(p.requires_grad for p in sub_submodule.parameters()):
                        continue

                    # Include class name in the logging key for better identification
                    class_name = sub_submodule.__class__.__name__
                    grad_norms[f"grad_norm/{name}/{submodule_name}/{sub_submodule_name}_{class_name}"] = (
                        self._calculate_grad_norm(sub_submodule.parameters())
                    )

        return grad_norms

    @staticmethod
    def _calculate_grad_norm(parameters):
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm**2
        return total_norm**0.5


class VAE(BaseModel):
    def __init__(
        self,
        # vae
        vae_model: TransformerVAE,
        vae_optimizer: Callable[[], Any],
        vae_scheduler: Callable[[int], float] | None = None,
        calculate_grad_norms: bool = False,
        # generation
        generation_args: DictConfig | None = None,
        inference_args: DictConfig | None = None,
        compile: bool = False,
        compile_mode: str = "default",
    ):
        super().__init__()

        self.vae_model = vae_model
        self.model_is_compiled = compile
        self.compile_mode = compile_mode

        self.vae_scheduler = vae_scheduler
        self.vae_optimizer = vae_optimizer

        self.metric_fns = REGRESSION_METRICS

        self.calculate_grad_norms = calculate_grad_norms

        self.generation_args = generation_args
        self.inference_args = inference_args

    def on_fit_start(self) -> None:
        if self.model_is_compiled:
            logger.info(f"Compiling model with {self.compile_mode} mode.")
            self.vae_model_compiled = torch.compile(
                self.vae_model, mode=self.compile_mode, dynamic=True, fullgraph=False
            )

    def configure_optimizers(self):
        vae_params = [p for p in self.vae_model.parameters() if p.requires_grad]
        vae_config = (
            {"optimizer": self.vae_optimizer(vae_params)} if vae_params else {}
        )  # empty dict is the case when vae is frozen

        if self.vae_scheduler is not None and vae_config:
            vae_config["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(vae_config["optimizer"], self.vae_scheduler),
                "interval": "step",
            }

        return vae_config

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
        counts_subset: torch.Tensor | None = None,
        genes_subset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model_is_compiled:
            if not hasattr(self, "vae_model_compiled"):
                logger.info(f"Compiling model with {self.compile_mode} mode during forward pass.")
                self.vae_model_compiled = torch.compile(
                    self.vae_model, mode="reduce-overhead", dynamic=False, fullgraph=True
                )
            logger.debug("Using compiled VAE model")
            return self.vae_model_compiled(counts, genes, library_size, counts_subset, genes_subset)
        else:
            logger.debug("Using non-compiled VAE model")
            return self.vae_model(counts, genes, library_size, counts_subset, genes_subset)

    def loss(
        self,
        counts: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
    ) -> dict[str, Any]:
        recon_loss = -log_nb_positive(counts, mu, theta)
        output = {
            LossEnum.LLH_LOSS.value: recon_loss.sum(dim=1).mean(),
        }
        return output

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        counts, genes = batch[ModelEnum.COUNTS.value], batch[ModelEnum.GENES.value]
        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)
        library_size = batch[ModelEnum.LIBRARY_SIZE.value]

        mu, theta, _ = self.forward(
            counts=counts,
            genes=genes,
            library_size=library_size,
            counts_subset=counts_subset,
            genes_subset=genes_subset,
        )

        loss_output = self.loss(
            counts=counts,
            mu=mu,
            theta=theta,
        )

        self.log("train_theta", theta.mean(), on_step=True, on_epoch=True, sync_dist=True)

        loss = sum(loss_output.values())
        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        for k, v in loss_output.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True, sync_dist=True)

        if self.calculate_grad_norms:
            modules = {
                "encoder": self.vae_model.encoder,
                "decoder": self.vae_model.decoder,
                "encoder_head": self.vae_model.encoder_head,
                "decoder_head": self.vae_model.decoder_head,
            }
            if hasattr(self.vae_model, "input_layer"):
                modules["input_layer"] = self.vae_model.input_layer
            grad_norms = self._compute_gradient_norms(modules=modules)
            self.log_dict(grad_norms, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def shared_step(self, batch, batch_idx, stage: str, ema: bool = False) -> dict[str, Any]:
        counts, genes = batch[ModelEnum.COUNTS.value], batch[ModelEnum.GENES.value]
        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)
        library_size = batch[ModelEnum.LIBRARY_SIZE.value]

        mu, theta, _ = self.vae_model(
            counts,
            genes,
            library_size,
            counts_subset,
            genes_subset,
        )

        loss_output = self.loss(
            counts=counts,
            mu=mu,
            theta=theta,
        )

        loss = sum(loss_output.values())
        metrics = {}
        metrics[f"{stage}_loss"] = loss
        for k, v in loss_output.items():
            metrics[f"{stage}_{k}"] = v

        counts_pred = NegativeBinomialSCVI(mu=mu, theta=theta).sample()

        counts_pred_scaled = torch.log1p((counts_pred / counts_pred.sum(dim=1, keepdim=True)) * 10_000)
        counts_true_scaled = torch.log1p((counts / counts.sum(dim=1, keepdim=True)) * 10_000)

        counts_pred_zeros = (counts_pred == 0).float()
        counts_true_zeros = (counts == 0).float()

        metrics[f"{stage}_zeros_accuracy"] = (counts_pred_zeros == counts_true_zeros).float().mean()

        for k, fn in self.metric_fns.items():
            output = fn(counts_pred_scaled, counts_true_scaled)
            metrics[f"{stage}_{k}"] = torch.nanmean(output)

        return metrics

    @torch.no_grad()
    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        outputs = self.inference(batch)
        batch.update(outputs)
        return tree_map(lambda x: x.cpu(), batch)

    @torch.no_grad()
    def sample(
        self,
        library_size: torch.Tensor,
        genes: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Sampling is not implemented for VAE")

    @torch.no_grad()
    def inference(
        self,
        batch: dict[str, torch.Tensor],
        n_samples: int | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        counts = batch[ModelEnum.COUNTS.value]
        genes = batch[ModelEnum.GENES.value]
        library_size = batch[ModelEnum.LIBRARY_SIZE.value]
        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)

        _, _, z = self.forward(
            counts=counts,
            genes=genes,
            library_size=library_size,
            counts_subset=counts_subset,
            genes_subset=genes_subset,
        )
        output: dict[str, torch.Tensor] = {
            "z_mean_flat": z.flatten(start_dim=1),
        }
        return output


class LatentDiffusion(BaseModel):
    def __init__(
        self,
        # vae
        vae_model: TransformerVAE,
        vae_optimizer: Callable[[], Any],
        # diffusion
        diffusion_model: DiT,
        transport: Transport,
        diffusion_scheduler: Callable[[int], float],
        diffusion_optimizer: Callable[[], Any],
        # more vae
        vae_scheduler: Callable[[int], float] | None = None,
        # ema
        ema_decay: float = 0.999,
        ema_update_every: int = 1,
        update_after_step: int = 1000,
        allow_different_devices: bool = True,
        use_foreach: bool = True,
        calculate_grad_norms: bool = False,
        # generation
        generation_args: DictConfig | None = None,
        inference_args: DictConfig | None = None,
        vae_as_tokenizer: DictConfig | None = None,
        # generation evaluation
        eval_generation: DictConfig = DictConfig({"enabled": False}),
        compile: bool = False,
        compile_mode: str = "default",
    ):
        super().__init__()

        self.vae_model = vae_model

        self.vae_scheduler = vae_scheduler
        self.vae_optimizer = vae_optimizer

        self.metric_fns = REGRESSION_METRICS
        self.mmd_metric_fns = MMD_METRICS
        self.wasserstein_metric_fns = WASSERSTEIN_METRICS
        self.r2_metric_fns = R2_METRICS

        self.generation_args = generation_args
        self.inference_args = inference_args

        self.diffusion_scheduler = diffusion_scheduler
        self.diffusion_optimizer = diffusion_optimizer

        self.vae_as_tokenizer = vae_as_tokenizer
        if self.vae_as_tokenizer is not None and not getattr(self.vae_as_tokenizer, "train", False):
            logger.info("VAE model is frozen")
            self.freeze()
            self.vae_model.eval()

        self.diffusion_model = diffusion_model

        self.model_is_compiled = compile
        self.compile_mode = compile_mode

        self.transport = transport
        self.transport_sampler = Sampler(self.transport)
        self.mse_loss = nn.MSELoss()

        self.ema_model = EMA(
            model=self.diffusion_model,
            beta=ema_decay,  # exponential moving average factor
            update_every=ema_update_every,  # how often to update
            allow_different_devices=allow_different_devices,
            use_foreach=use_foreach,
            update_after_step=update_after_step,
        )
        self.mmd_metric_fns = MMD_METRICS
        self.calculate_grad_norms = calculate_grad_norms

        # Initialize attributes for generation evaluation
        self.eval_generation: DictConfig = eval_generation
        self.accumulated_generated_batches: list[torch.Tensor] = []
        self.accumulated_samples = 0  # number of samples accumulated for generation evaluation
        self.is_generation_eval_epoch = False  # used in on_validation*** to decide if it is a generation eval epoch

    def on_fit_start(self) -> None:
        if self.model_is_compiled:
            logger.info(f"Compiling model with {self.compile_mode} mode.")
            self.vae_model_compiled = torch.compile(
                self.vae_model, mode=self.compile_mode, dynamic=True, fullgraph=False
            )
            self.diffusion_model_compiled = torch.compile(
                self.diffusion_model, mode=self.compile_mode, dynamic=True, fullgraph=False
            )

    def _sample_log_size_factors(self, condition: dict[str, torch.Tensor] | None, batch_size: int) -> torch.Tensor:
        """Sample log size factors using joint or independent condition stats.

        Behavior:
        - If `self.diffusion_model.condition_strategy == "joint"` and the vocabulary
          encoder provides `joint_idx_2_classes` and a valid `joint_key` present in
          both `mu_size_factor` and `sd_size_factor`, build a joint class key per
          sample and draw from the corresponding Normal distribution.
        - Otherwise, fall back to independent sampling based on a single condition key.
          The key is resolved by `vocab.size_factor_condition_key` if available,
          otherwise inferred from the intersection of `condition.keys()` with
          `mu_size_factor`/`sd_size_factor` keys.

        If statistics are missing, return zeros for the affected samples and log a
        warning once, to keep generation running.
        """
        vocab_encoder = self.trainer.datamodule.vocabulary_encoder
        mu_size_factor = getattr(vocab_encoder, "mu_size_factor", None)
        sd_size_factor = getattr(vocab_encoder, "sd_size_factor", None)

        log_size_factors = torch.zeros(batch_size, device=self.device)

        # Early exit when stats or condition are unavailable
        if condition is None or mu_size_factor is None or sd_size_factor is None:
            return log_size_factors

        # Decide whether to use joint sampling
        use_joint = False
        joint_idx_2_classes = getattr(vocab_encoder, "joint_idx_2_classes", None)
        joint_key = getattr(vocab_encoder, "joint_key", None)
        if getattr(self.diffusion_model, "condition_strategy", None) == "joint" and joint_idx_2_classes is not None:
            # Safe casts after early None-check above
            mu_map = cast(dict, mu_size_factor)
            sd_map = cast(dict, sd_size_factor)
            if joint_key is not None and joint_key in mu_map and joint_key in sd_map:
                use_joint = True

        if use_joint:
            components = getattr(vocab_encoder, "joint_components", None)
            if components is not None:
                component_keys = [k for k in components if k in condition]
            else:
                component_keys = list(condition.keys())

            # Validate lengths for all component keys
            for k in component_keys:
                if len(condition[k]) != batch_size:
                    if not hasattr(self, "_warned_joint_len_mismatch"):
                        logger.warning(
                            "Length mismatch for joint components; expected %d, got different sizes. Using zeros.",
                            batch_size,
                        )
                        self._warned_joint_len_mismatch = True
                    return log_size_factors

            mu_map = cast(dict, mu_size_factor)
            sd_map = cast(dict, sd_size_factor)
            for bch_idx in range(batch_size):
                indices = [int(condition[k][bch_idx].item()) for k in component_keys]
                key = "_".join(str(i) for i in indices)
                if key not in joint_idx_2_classes:
                    if not hasattr(self, "_warned_missing_joint_key"):
                        logger.warning(
                            "Joint key '%s' not found in vocabulary_encoder.joint_idx_2_classes; using zero.", key
                        )
                        self._warned_missing_joint_key = True
                    continue
                class_idx = joint_idx_2_classes[key]
                mu_vec = cast(dict, mu_map[joint_key])  # type: ignore[index]
                sd_vec = cast(dict, sd_map[joint_key])  # type: ignore[index]
                mean_val = mu_vec.get(class_idx)
                std_val = sd_vec.get(class_idx)
                if mean_val is None or std_val is None:
                    if not hasattr(self, "_warned_missing_stats"):
                        logger.warning("Missing mean/std for joint size factor; using zero for affected samples.")
                        self._warned_missing_stats = True
                    continue
                log_size_factors[bch_idx] = Normal(loc=mean_val, scale=std_val).sample()
            return log_size_factors

        # Independent sampling path
        size_factor_condition_key = getattr(vocab_encoder, "size_factor_condition_key", None)
        selected_key: str | None = None
        if (
            size_factor_condition_key
            and size_factor_condition_key in condition
            and size_factor_condition_key in mu_size_factor
            and size_factor_condition_key in sd_size_factor
        ):
            selected_key = size_factor_condition_key
        else:
            cond_keys = set(condition.keys())
            mu_keys = set(mu_size_factor.keys())
            sd_keys = set(sd_size_factor.keys())
            inter = sorted(cond_keys & mu_keys & sd_keys)
            if inter:
                selected_key = inter[0]
                if not hasattr(self, "_warned_inferred_condition_key"):
                    logger.warning("Inferred size-factor condition key '%s' for independent sampling.", selected_key)
                    self._warned_inferred_condition_key = True
            else:
                if not hasattr(self, "_warned_no_condition_key"):
                    logger.warning("No matching condition key found in mu/sd for size-factor sampling; using zeros.")
                    self._warned_no_condition_key = True
                return log_size_factors

        assert selected_key is not None
        labels = condition[selected_key]
        if len(labels) != batch_size:
            raise ValueError(f"Condition '{selected_key}' length ({len(labels)}) must match batch size ({batch_size})")
        mu_map = cast(dict, mu_size_factor)
        sd_map = cast(dict, sd_size_factor)
        for i in range(batch_size):
            class_idx = int(labels[i].item())
            mu_vec = cast(dict, mu_map[selected_key])
            sd_vec = cast(dict, sd_map[selected_key])
            mean_val = mu_vec.get(class_idx)
            std_val = sd_vec.get(class_idx)
            if mean_val is None or std_val is None:
                if not hasattr(self, "_warned_missing_stats"):
                    logger.warning("Missing mean/std for independent size factor; using zero for affected samples.")
                    self._warned_missing_stats = True
                continue
            log_size_factors[i] = Normal(loc=mean_val, scale=std_val).sample()
        return log_size_factors

    def configure_optimizers(self):
        diffusion_params = [p for p in self.diffusion_model.parameters() if p.requires_grad]
        diffusion_config = {"optimizer": self.diffusion_optimizer(diffusion_params)}

        if self.diffusion_scheduler is not None:
            diffusion_config["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(diffusion_config["optimizer"], self.diffusion_scheduler),
                "interval": "step",
            }

        return diffusion_config

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        counts_subset: torch.Tensor | None = None,
        genes_subset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        if self.model_is_compiled:
            z = self.vae_model_compiled.encode(
                counts,
                genes,
                counts_subset,
                genes_subset,
            )
        else:
            z = self.vae_model.encode(
                counts,
                genes,
                counts_subset,
                genes_subset,
            )
        return z

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        counts = batch[ModelEnum.COUNTS.value]
        genes = batch[ModelEnum.GENES.value]
        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)
        exclude_keys = (ModelEnum.COUNTS.value, ModelEnum.GENES.value, ModelEnum.LIBRARY_SIZE.value)

        z = self.forward(
            counts=counts,
            genes=genes,
            counts_subset=counts_subset,
            genes_subset=genes_subset,
        )

        # Prepare all available conditions for CFG dropout (similar to SiT approach)
        condition_keys = [k for k in batch.keys() if k not in exclude_keys]
        condition = {k: batch[k] for k in condition_keys}
        model_kwargs = {"condition": condition}

        if self.model_is_compiled:
            loss_dict = self.transport.training_losses(self.diffusion_model_compiled, z, model_kwargs)
        else:
            loss_dict = self.transport.training_losses(self.diffusion_model, z, model_kwargs)

        loss_output = {"train_loss": loss_dict["loss"].mean()}

        self.log("train_loss", loss_output["train_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        if self.calculate_grad_norms:
            grad_norms = self._compute_gradient_norms({"diffusion": self.diffusion_model})
            self.log_dict(grad_norms, on_step=True, on_epoch=True, sync_dist=True)

        return loss_output["train_loss"]

    def freeze(self):
        """Freeze the vae model parameters"""
        logger.info("Freezing the vae model parameters")
        for param in self.vae_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def shared_step(self, batch, batch_idx, stage: str, ema: bool = False) -> dict[str, Any]:
        counts = batch[ModelEnum.COUNTS.value]
        genes = batch[ModelEnum.GENES.value]
        # counts_subset = batch[ModelEnum.COUNTS_SUBSET.value]
        # genes_subset = batch[ModelEnum.GENES_SUBSET.value]

        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)
        exclude_keys = (
            ModelEnum.COUNTS.value,
            ModelEnum.GENES.value,
            ModelEnum.COUNTS_SUBSET.value,
            ModelEnum.GENES_SUBSET.value,
            ModelEnum.LIBRARY_SIZE.value,
        )
        condition = {k: batch[k] for k in batch if k not in exclude_keys}

        model = self.ema_model if ema else self.diffusion_model
        stage = stage + "_ema" if ema else stage

        z = self.forward(
            counts=counts,
            genes=genes,
            counts_subset=counts_subset,
            genes_subset=genes_subset,
        )
        loss_dict = self.transport.training_losses(model, z, {"condition": condition})

        metrics = {}
        metrics[f"{stage}_loss"] = loss_dict["loss"].mean()
        metrics[f"{stage}_{LossEnum.DIFF_LOSS.value}"] = loss_dict["loss"].mean()
        return metrics

    @torch.no_grad()
    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor] | None:
        if self.generation_args is not None:
            generation_kwargs = {str(k): v for k, v in self.generation_args.items()} if self.generation_args else {}
            guidance_weight = generation_kwargs.get("guidance_weight", None)
            timesteps = generation_kwargs.get("timesteps", 50)

            exclude_keys = (
                ModelEnum.COUNTS.value,
                ModelEnum.GENES.value,
                ModelEnum.COUNTS_SUBSET.value,
                ModelEnum.GENES_SUBSET.value,
                ModelEnum.LIBRARY_SIZE.value,
            )

            condition = {k: v for k, v in batch.items() if k not in exclude_keys}
            size_factors = batch[ModelEnum.LIBRARY_SIZE.value]
            genes = batch[ModelEnum.GENES.value]

            nb_outputs, z_outputs = self.sample(
                condition=condition,
                guidance_weight=guidance_weight,
                batch_size=len(size_factors),
                genes=genes,
                timesteps=timesteps,
            )
            batch_size_single = len(size_factors)
            # first half is unconditional, second half is conditional
            batch[f"{ModelEnum.COUNTS.value}_generated_unconditional"] = nb_outputs[:batch_size_single]
            batch[f"{ModelEnum.COUNTS.value}_generated_conditional"] = nb_outputs[batch_size_single:]
            batch["z_generated_unconditional"] = z_outputs[:batch_size_single].flatten(start_dim=1)
            batch["z_generated_conditional"] = z_outputs[batch_size_single:].flatten(start_dim=1)
            return tree_map(lambda x: x.cpu(), batch)
        elif self.inference_args is not None:
            from scldm._train_utils import create_anndata_from_inference_output

            logger.info("Running inference")
            encode_kwargs = {str(k): v for k, v in self.inference_args.items()} if self.inference_args else {}

            inference_outputs: dict[str, torch.Tensor] = self.inference(
                batch=batch,
                **encode_kwargs,
            )
            excluded_keys = (
                ModelEnum.COUNTS.value,
                ModelEnum.GENES.value,
                ModelEnum.LIBRARY_SIZE.value,
                ModelEnum.COUNTS_SUBSET.value,
                ModelEnum.GENES_SUBSET.value,
            )
            inference_outputs.update({k: batch[k].cpu().numpy() for k in batch if k not in excluded_keys})
            adata = create_anndata_from_inference_output(inference_outputs, self.trainer.datamodule)
            return adata
        else:
            raise ValueError("No generation or encode args provided")

    @torch.no_grad()
    def sample(
        self,
        condition: dict[str, torch.Tensor] | None,
        guidance_weight: dict[str, float] | None,
        batch_size: int,
        genes: torch.Tensor,
        timesteps: int = 50,
    ) -> torch.Tensor:
        # Validate inputs
        if len(genes) != batch_size:
            raise ValueError(f"genes batch dimension ({genes.shape[0]}) must match batch_size ({batch_size})")

        if condition is not None:
            for key, values in condition.items():
                if len(values) != batch_size:
                    raise ValueError(f"Condition '{key}' length ({len(values)}) must match batch size ({batch_size})")

        # Sample size factors from normal distributions based on condition labels
        size_factors = self._sample_log_size_factors(condition, batch_size)

        # Initial latent noise
        z = torch.randn(
            (batch_size, self.diffusion_model.seq_len, self.vae_model.encoder.latent_embedding),
            device=self.device,
        )

        sample_fn = self.transport_sampler.sample_ode()

        # Validate guidance_weight keys match condition keys (only if both are not None)
        if guidance_weight is not None and condition is not None:
            assert set(guidance_weight.keys()) == set(condition.keys()), (
                f"Guidance weight keys {set(guidance_weight.keys())} must match condition keys {set(condition.keys())}"
            )

        z_cfg = torch.cat([z, z], dim=0)

        # Duplicate conditions for CFG
        condition_cfg = {}
        if condition is not None:
            for key, values in condition.items():
                condition_cfg[key] = torch.cat([values, values], dim=0)

        model_fn = lambda x, t, **kwargs: self.diffusion_model.forward_with_cfg(
            x, t, **kwargs, cfg_scale=guidance_weight
        )
        samples = sample_fn(z_cfg, model_fn, **{"condition": condition_cfg})[-1]

        genes = torch.cat([genes, genes], dim=0)

        size_factors_actual = torch.exp(size_factors).view(-1, 1)  # shape: (batch_size, 1)
        size_factors_cfg = torch.cat([size_factors_actual, size_factors_actual], dim=0)
        nb = self.vae_model.decode(samples, genes, size_factors_cfg)
        return nb.sample(), samples

    @torch.no_grad()
    def inference(
        self,
        batch: dict[str, torch.Tensor],
        n_samples: int,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        counts = batch[ModelEnum.COUNTS.value]
        genes = batch[ModelEnum.GENES.value]
        library_size = batch[ModelEnum.LIBRARY_SIZE.value]
        counts_subset = batch[ModelEnum.COUNTS_SUBSET.value]
        genes_subset = batch[ModelEnum.GENES_SUBSET.value]

        mu, theta, z = self.vae_model.forward(
            counts=counts,
            genes=genes,
            library_size=library_size,
            counts_subset=counts_subset,
            genes_subset=genes_subset,
        )
        output: dict[str, torch.Tensor] = {
            "z_mean_flat": z.flatten(start_dim=1),
        }
        return output

    def on_validation_epoch_start(self) -> None:
        """Check if this is a generation evaluation epoch and initialize accumulation."""
        super().on_validation_start()

        if (
            self.eval_generation.enabled
            and self.current_epoch % self.eval_generation.freq == 0
            and self.current_epoch > self.eval_generation.warmup_epochs
            and self.current_epoch > 0
        ):
            self.is_generation_eval_epoch = True
            self.accumulated_generated_batches = []
            if dist.is_initialized():
                rank = dist.get_rank()
                logger.info(f"Rank {rank} - Starting generation evaluation at epoch {self.current_epoch}")
        else:
            self.is_generation_eval_epoch = False

    def validation_step(self, batch: dict[str, torch.Tensor | dict[str, torch.Tensor]], batch_idx: int) -> None:
        """Override validation_step to accumulate batches during generation evaluation epochs."""
        super().validation_step(batch, batch_idx)

        if self.is_generation_eval_epoch:
            if self.accumulated_samples < self.eval_generation.sample_size:
                # Cast batch to the expected type for sample method
                timesteps = self.generation_args.get("timesteps", 50)
                genes = batch[ModelEnum.GENES.value]
                logger.info("Generating samples.")
                outputs = self.sample(
                    condition=None,
                    guidance_weight=None,
                    batch_size=len(batch[ModelEnum.COUNTS.value]),
                    genes=genes,
                    timesteps=timesteps,
                )
                batch[f"{ModelEnum.COUNTS.value}_generated"] = outputs
                self.accumulated_generated_batches.append(tree_map(lambda x: x.cpu(), batch))
                self.accumulated_samples += len(batch[ModelEnum.COUNTS.value])

    def on_validation_epoch_end(self) -> None:
        """Process accumulated batches for generation evaluation."""
        if self.is_generation_eval_epoch and len(self.accumulated_generated_batches) > 0:
            # Concatenate all accumulated batches
            counts = torch.cat([b[ModelEnum.COUNTS.value] for b in self.accumulated_generated_batches], dim=0)
            counts_generated = torch.cat(
                [b[f"{ModelEnum.COUNTS.value}_generated"] for b in self.accumulated_generated_batches], dim=0
            )
            library_size = torch.cat(
                [b[ModelEnum.LIBRARY_SIZE.value] for b in self.accumulated_generated_batches], dim=0
            )
            counts_generated_scaled = torch.log1p((counts_generated / library_size) * 10_000)
            counts_true_scaled = torch.log1p((counts / library_size) * 10_000)
            logger.info("Computing generation evaluation metrics.")
            for k, fn in self.mmd_metric_fns.items():
                if "counts" in k:
                    mmd = fn(counts_true_scaled, counts_generated_scaled)
                else:
                    mmd = fn(counts, counts_generated)
                self.log(
                    f"generation_eval/{k}",
                    torch.nanmean(mmd) if "counts" in k else torch.nanmean(mmd),
                    on_epoch=True,
                    sync_dist=True,
                )
            for k, fn in self.wasserstein_metric_fns.items():
                wdist = fn(counts_true_scaled, counts_generated_scaled)
                self.log(
                    f"generation_eval/{k}",
                    wdist,
                    on_epoch=True,
                    sync_dist=True,
                )
            for k, fn in self.r2_metric_fns.items():
                r2 = fn(counts_true_scaled, counts_generated_scaled)
                self.log(
                    f"generation_eval/{k}",
                    r2,
                    on_epoch=True,
                    sync_dist=True,
                )

            self.log("generation_eval/total_samples", counts.shape[0], on_epoch=True, sync_dist=True)

            if dist.is_initialized():
                rank = dist.get_rank()
                logger.info(f"Rank {rank} - Generation evaluation completed with {counts.shape[0]} total samples")

            # Clear accumulated batches to free memory
            self.accumulated_generated_batches = []
            self.accumulated_samples = 0
            self.is_generation_eval_epoch = False


class VAEScvi(BaseModel):
    def __init__(
        self,
        # vae
        vae_model: TransformerVAE,
        vae_optimizer: Callable[[], Any],
        vae_scheduler: Callable[[int], float] | None = None,
        # loss
        kl_weight: float = 1.0,
        cr_weight: float = 0.0,
        masking_prop: float = 0.0,
        mask_token_idx: int = 0,
        # ema
        ema_decay: float = 0.999,
        ema_update_every: int = 1,
        update_after_step: int = 1000,
        allow_different_devices: bool = True,
        use_foreach: bool = True,
        calculate_grad_norms: bool = False,
        # generation
        generation_args: DictConfig | None = None,
        inference_args: DictConfig | None = None,
        compile: bool = False,
        compile_mode: str = "default",
    ):
        super().__init__()

        self.vae_model = vae_model

        self.model_is_compiled = compile
        self.compile_mode = compile_mode

        self.vae_scheduler = vae_scheduler
        self.vae_optimizer = vae_optimizer

        self.metric_fns = REGRESSION_METRICS

        self.kl_weight = kl_weight
        self.masking_prop = masking_prop
        self.cr_weight = cr_weight
        self.mask_token_idx = mask_token_idx
        if self.masking_prop > 0:
            assert cr_weight is not None, "cr_weight must be provided if masking_prop is greater than 0"
        self.calculate_grad_norms = calculate_grad_norms

        self.generation_args = generation_args
        self.inference_args = inference_args

    def on_fit_start(self) -> None:
        if self.model_is_compiled:
            logger.info(f"Compiling model with {self.compile_mode} mode.")
            self.vae_model_compiled = torch.compile(
                self.vae_model, mode=self.compile_mode, dynamic=True, fullgraph=False
            )

    def configure_optimizers(self):
        vae_params = [p for p in self.vae_model.parameters() if p.requires_grad]
        vae_config = (
            {"optimizer": self.vae_optimizer(vae_params)} if vae_params else {}
        )  # empty dict is the case when vae is frozen

        if self.vae_scheduler is not None and vae_config:
            vae_config["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(vae_config["optimizer"], self.vae_scheduler),
                "interval": "step",
            }

        return vae_config

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
        condition: dict[str, torch.Tensor] | None = None,
        counts_subset: torch.Tensor | None = None,
        genes_subset: torch.Tensor | None = None,
        masking_prop: float = 0.0,
        mask_token_idx: int = 0,
    ) -> tuple[Distribution, Distribution, torch.Tensor]:
        if self.model_is_compiled:
            return self.vae_model_compiled(
                counts, genes, library_size, condition, counts_subset, genes_subset, masking_prop, mask_token_idx
            )
        else:
            return self.vae_model(
                counts, genes, library_size, condition, counts_subset, genes_subset, masking_prop, mask_token_idx
            )

    def loss(
        self,
        counts: torch.Tensor,
        conditional_likelihood: Distribution,
        variational_posterior: Distribution,
        z_sample: torch.Tensor,
        conditional_likelihood_masked: Distribution | None = None,
        variational_posterior_masked: Distribution | None = None,
        z_sample_masked: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        recon_loss = -conditional_likelihood.log_prob(counts)
        kl_loss = self.kl_weight * (variational_posterior.log_prob(z_sample) - self.vae_model.prior.log_prob(z_sample))

        output = {
            LossEnum.LLH_LOSS.value: recon_loss.sum(dim=1).mean(),
            LossEnum.KL_LOSS.value: kl_loss.sum(dim=1).mean(),
        }

        for k, v in output.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN values detected in {k}")

        return output

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        counts, genes = batch[ModelEnum.COUNTS.value], batch[ModelEnum.GENES.value]
        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)
        library_size = batch[ModelEnum.LIBRARY_SIZE.value]
        exclude_keys = (
            ModelEnum.COUNTS.value,
            ModelEnum.GENES.value,
            ModelEnum.COUNTS_SUBSET.value,
            ModelEnum.GENES_SUBSET.value,
            ModelEnum.LIBRARY_SIZE.value,
        )
        condition = {k: batch[k] for k in batch if k not in exclude_keys}

        # Forward returns (conditional_likelihood, variational_posterior, z), we only need z
        conditional_likelihood, variational_posterior, z = self.forward(
            counts=counts,
            genes=genes,
            library_size=library_size,
            condition=condition,
            counts_subset=counts_subset,
            genes_subset=genes_subset,
        )
        conditional_likelihood_masked = None
        variational_posterior_masked = None
        z_masked = None

        loss_output = self.loss(
            counts=counts,
            conditional_likelihood=conditional_likelihood,
            variational_posterior=variational_posterior,
            z_sample=z,
            conditional_likelihood_masked=conditional_likelihood_masked,
            variational_posterior_masked=variational_posterior_masked,
            z_sample_masked=z_masked,
        )

        if hasattr(conditional_likelihood, "theta"):
            self.log("train_theta", conditional_likelihood.theta.mean(), on_step=True, on_epoch=True, sync_dist=True)
        if hasattr(conditional_likelihood, "scale") and conditional_likelihood.scale is not None:
            self.log("train_scale", conditional_likelihood.scale.mean(), on_step=True, on_epoch=True, sync_dist=True)

        loss = sum(loss_output.values())
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        for k, v in loss_output.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True, sync_dist=True)

        if self.calculate_grad_norms:
            modules = {
                "encoder": self.vae_model.encoder,
                "decoder": self.vae_model.decoder,
                "encoder_head": self.vae_model.encoder_head,
                "decoder_head": self.vae_model.decoder_head,
            }
            if hasattr(self.vae_model, "input_layer"):
                modules["input_layer"] = self.vae_model.input_layer
            grad_norms = self._compute_gradient_norms(modules=modules)
            self.log_dict(grad_norms, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def shared_step(self, batch, batch_idx, stage: str, ema: bool = False) -> dict[str, Any]:
        counts, genes = batch[ModelEnum.COUNTS.value], batch[ModelEnum.GENES.value]
        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)
        library_size = batch[ModelEnum.LIBRARY_SIZE.value]
        exclude_keys = (
            ModelEnum.COUNTS.value,
            ModelEnum.GENES.value,
            ModelEnum.COUNTS_SUBSET.value,
            ModelEnum.GENES_SUBSET.value,
            ModelEnum.LIBRARY_SIZE.value,
        )
        condition = {k: batch[k] for k in batch if k not in exclude_keys}

        model = self.vae_model
        model.eval()

        conditional_likelihood, variational_posterior, z = model(
            counts,
            genes,
            library_size,
            condition,
            counts_subset,
            genes_subset,
        )

        loss_output = self.loss(
            counts=counts,
            conditional_likelihood=conditional_likelihood,
            variational_posterior=variational_posterior,
            z_sample=z,
        )

        loss = sum(loss_output.values())
        metrics = {}
        metrics[f"{stage}_loss"] = loss
        for k, v in loss_output.items():
            metrics[f"{stage}_{k}"] = v

        counts_pred = conditional_likelihood.sample()

        real_library_size = counts.sum(dim=1, keepdim=True)
        counts_pred_scaled = torch.log1p((counts_pred / real_library_size) * 10_000)
        counts_true_scaled = torch.log1p((counts / real_library_size) * 10_000)

        counts_pred_zeros = (counts_pred == 0).float()
        counts_true_zeros = (counts == 0).float()

        metrics[f"{stage}_zeros_accuracy"] = (counts_pred_zeros == counts_true_zeros).float().mean()

        for k, fn in self.metric_fns.items():
            output = fn(counts_pred_scaled, counts_true_scaled)
            metrics[f"{stage}_{k}"] = torch.nanmean(output)

        return metrics

    @torch.no_grad()
    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        from scldm._train_utils import create_anndata_from_inference_output

        outputs = self.inference(batch)
        batch.update(outputs)
        if self.generation_args is not None:
            return tree_map(lambda x: x.cpu(), batch)
        else:
            return create_anndata_from_inference_output(tree_map(lambda x: x.cpu(), batch), self.trainer.datamodule)
        # return tree_map(
        #     lambda x: x.cpu() if isinstance(x, torch.Tensor) else torch.from_numpy(x).cpu(),
        #     batch
        # )

    @torch.no_grad()
    def sample(
        self,
        library_size: torch.Tensor,
        genes: torch.Tensor,
    ) -> torch.Tensor:
        z_sample_prior = self.vae_model.prior.sample(n_samples=len(library_size))
        nb = self.vae_model.decode(z_sample_prior, genes, library_size, None)
        return nb.sample()

    @torch.no_grad()
    def inference(
        self,
        batch: dict[str, torch.Tensor],
        n_samples: int | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        counts = batch[ModelEnum.COUNTS.value]
        genes = batch[ModelEnum.GENES.value]
        library_size = batch[ModelEnum.LIBRARY_SIZE.value]
        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)
        exclude_keys = (
            ModelEnum.COUNTS.value,
            ModelEnum.GENES.value,
            ModelEnum.LIBRARY_SIZE.value,
            ModelEnum.COUNTS_SUBSET.value,
            ModelEnum.GENES_SUBSET.value,
        )
        condition = {k: batch[k] for k in batch if k not in exclude_keys}

        nb, variational_posterior, _ = self.forward(
            counts=counts,
            genes=genes,
            library_size=library_size,
            condition=condition,
            counts_subset=counts_subset,
            genes_subset=genes_subset,
        )
        z = variational_posterior.sample((n_samples,))
        output: dict[str, torch.Tensor] = {
            "z_mean_flat": z.flatten(start_dim=1),
        }
        return output
