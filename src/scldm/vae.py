import torch
import torch.nn as nn
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
        )  # B, S, E
        h_z = self.encoder(genes_counts_embedding)  # B, M, E
        genes_for_decoder = (
            genes if isinstance(self.decoder.gene_embedding, nn.Embedding) else self.input_layer.gene_embedding(genes)
        )  # B, S, E
        h_x = self.decoder(h_z, genes_for_decoder)  # B, S, E
        mu, theta = self.decoder_head(h_x, genes, library_size)  # B, S, 1
        return mu, theta, h_z

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
