import torch
import torch.nn as nn
from einops import rearrange
from scvi.distributions import NegativeBinomial as NegativeBinomialSCVI
from torch.distributions import Distribution

from scldm.layers import InputTransformerVAE
from scldm.nnets import Decoder, DecoderScvi, Encoder, EncoderScvi
from scldm.stochastic_layers import (
    GaussianLinearLayer,
    NegativeBinomialLinearLayer,
    NegativeBinomialTransformerLayer,
)


class TransformerVAE(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        decoder_head: NegativeBinomialTransformerLayer,
        input_layer: InputTransformerVAE,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_head = decoder_head
        self.input_layer = input_layer

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
        counts_subset: torch.Tensor | None = None,
        genes_subset: torch.Tensor | None = None,
    ) -> tuple[Distribution, torch.Tensor]:
        genes_counts_embedding = self.input_layer(
            counts_subset,
            genes_subset,
            # counts_subset if counts_subset is not None else counts,
            # genes_subset if genes_subset is not None else genes,
        )  # B, S, E
        h_z = self.encoder(genes_counts_embedding)  # B, M, E
        genes_for_decoder = (
            genes if isinstance(self.decoder.gene_embedding, nn.Embedding) else self.input_layer.gene_embedding(genes)
        )  # B, S, E
        h_x = self.decoder(h_z, genes_for_decoder)  # B, S, E
        mu, theta = self.decoder_head(h_x, genes, library_size)  # B, S, 1
        return mu, theta, h_z

    def interpolate_forward(
        self,
        control_cells_counts: torch.Tensor,
        cytokine_cells_counts: torch.Tensor,
        genes: torch.Tensor,
        control_library_size: torch.Tensor,
        cytokine_library_size: torch.Tensor,
        condition: dict[str, torch.Tensor] | None = None,
        masking_prop: float = 0.0,
        mask_token_idx: int = 0,
    ) -> tuple[torch.Tensor, Distribution, torch.Tensor]:
        # Combine control and cytokine counts for processing
        # Ensure all tensors are on the same device

        # get device from encoder
        device = list(self.encoder.parameters())[0].device.type
        control_cells_counts = control_cells_counts.to(device)
        cytokine_cells_counts = cytokine_cells_counts.to(device)
        genes = genes.to(device)
        control_library_size = control_library_size.to(device)
        cytokine_library_size = cytokine_library_size.to(device)

        combined_counts = torch.cat([control_cells_counts, cytokine_cells_counts], dim=0)
        # combined_library_size = torch.cat([control_library_size, cytokine_library_size], dim=0)
        genes_counts_embedding, condition_embedding = self.input_layer(
            combined_counts,
            genes,
            condition,
            masking_prop,
            mask_token_idx,
        )

        gene_masking = (combined_counts > 0).bool().contiguous()
        h_z, _ = self.encoder(genes_counts_embedding, gene_masking=gene_masking)
        variational_posterior = self.encoder_head(h_z)
        loc = getattr(variational_posterior, "loc", None)
        scale = getattr(variational_posterior, "scale", None)
        if loc is not None and scale is not None:
            eps = torch.randn_like(loc)
            z = loc + eps * scale
        else:
            z = variational_posterior.rsample()

        # Simple interpolation: mix current z with random z from same batch

        genes_decoder = (
            genes if isinstance(self.decoder.gene_embedding, nn.Embedding) else self.input_layer.gene_embedding(genes)
        )

        # interpolate between control and cytokine counts
        # get control z and cytokine z
        control_z = z[: len(control_cells_counts)]
        cytokine_z = z[len(control_cells_counts) :]
        # interpolate between control and cytokine z
        # get 10 alphas
        alphas = torch.linspace(0, 1, 10, device=z.device)

        # interpolate between control and cytokine z
        # unsqueeze to get correct shapes
        control_z = control_z.unsqueeze(1)
        cytokine_z = cytokine_z.unsqueeze(1)
        alphas = alphas.unsqueeze(1)
        alphas = alphas.unsqueeze(2).unsqueeze(0)

        # interpolate between control and cytokine z
        z = (1 - alphas) * control_z + alphas * cytokine_z

        # interpolate library size count as well with the same alphas

        alphas = alphas.squeeze(-1).squeeze(-1)
        interpolated_library_size = (1 - alphas) * control_library_size + alphas * cytokine_library_size

        z = rearrange(z, "b s i d -> (b s) i d")
        interpolated_library_size = rearrange(interpolated_library_size, "b s -> (b s)").unsqueeze(-1)

        genes_decoder = genes_decoder[: len(control_cells_counts)].repeat(10, 1, 1)

        h_x = self.decoder(z, genes_decoder, condition_embedding)

        genes = genes[: len(control_cells_counts)].repeat(10, 1)

        conditional_likelihood = self.decoder_head(h_x, genes, interpolated_library_size)
        reconstructed_counts = conditional_likelihood.mu
        return reconstructed_counts, variational_posterior, z

    def encode(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        counts_subset: torch.Tensor | None = None,
        genes_subset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        genes_counts_embedding = self.input_layer(
            counts_subset if counts_subset is not None else counts,
            genes_subset if genes_subset is not None else genes,
        )
        return self.encoder(genes_counts_embedding)

    def decode(
        self,
        z: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
        condition: dict[str, torch.Tensor] | None = None,
    ) -> torch.distributions.Distribution:
        genes_for_decoder = (
            genes if isinstance(self.decoder.gene_embedding, nn.Embedding) else self.input_layer.gene_embedding(genes)
        )
        h_x = self.decoder(z, genes_for_decoder, condition)
        mu, theta = self.decoder_head(h_x, genes, library_size)
        return NegativeBinomialSCVI(mu=mu, theta=theta)


class ScviVAE(nn.Module):
    def __init__(
        self,
        encoder: EncoderScvi,
        encoder_head: GaussianLinearLayer,
        decoder: DecoderScvi,
        decoder_head: NegativeBinomialLinearLayer,
        prior: Distribution,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_head = encoder_head
        self.decoder = decoder
        self.decoder_head = decoder_head
        self.prior = prior

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
        h_z, _ = self.encoder(counts)
        variational_posterior = self.encoder_head(h_z)
        loc = getattr(variational_posterior, "loc", None)
        scale = getattr(variational_posterior, "scale", None)
        if loc is not None and scale is not None:
            eps = torch.randn_like(loc)
            z = loc + eps * scale
        else:
            z = variational_posterior.rsample()
        h_x = self.decoder(z)
        conditional_likelihood = self.decoder_head(h_x, None, library_size)
        return conditional_likelihood, variational_posterior, z

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
