from typing import Literal

import torch
import torch.nn as nn

from scldm.layers import (
    Block,
    CrossAttentionBlock,
    FinalLayerDit,
    TimestepEmbedder,
    get_1d_sincos_pos_embed,
)

#############################
#        Like in scVI       #
#############################


class EncoderScvi(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_hidden: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.extend(
                [
                    nn.Linear(n_genes if i == 0 else n_hidden, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.encoder_mlp = nn.Sequential(*layers)
        self.latent_embedding = 0

    def forward(self, x: torch.Tensor, genes: torch.Tensor | None = None) -> tuple[torch.Tensor, None]:
        x = torch.log1p(x)
        x = self.encoder_mlp(x)
        return x, None


class DecoderScvi(nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_hidden: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.extend(
                [
                    nn.Linear(n_latent if i == 0 else n_hidden, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.decoder_mlp = nn.Sequential(*layers)
        self.last_embedding = None

    def forward(self, x: torch.Tensor, condition: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        x = self.decoder_mlp(x)
        return x


#################################
#         All Transformer       #
#################################


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer: int,
        n_inducing_points: int,
        n_embed: int,
        n_embed_latent: int,
        n_head: int,
        n_head_cross: int,
        dropout: float,
        bias: bool,
        multiple_of: int,
        layernorm_eps: float,
        norm_layer: str,
        positional_encoding: bool = False,
    ):
        super().__init__()

        self.latent_embedding = n_embed_latent
        self.latent_dim = n_inducing_points
        self.encoder_layers = nn.ModuleList([])
        self.pos_embed: torch.Tensor | None
        if positional_encoding:
            self.pos_embed = nn.Parameter(torch.zeros(1, n_inducing_points, n_embed), requires_grad=False)
        else:
            self.pos_embed = None

        self.ca_layer = CrossAttentionBlock(
            n_embed=n_embed,
            n_inducing_points=n_inducing_points,
            n_head=n_head_cross,
            dropout=dropout,
            bias=bias,
            norm_layer=norm_layer,
            multiple_of=multiple_of,
            layernorm_eps=layernorm_eps,
        )

        for _ in range(n_layer):
            self.encoder_layers.append(
                Block(
                    n_embed=n_embed,
                    n_head=n_head,
                    dropout=dropout,
                    bias=bias,
                    norm_layer=norm_layer,
                    multiple_of=multiple_of,
                    layernorm_eps=layernorm_eps,
                )
            )

        self.encoder_latent_input = nn.Sequential(
            nn.Linear(n_embed, n_embed_latent, bias=bias),
            nn.LayerNorm(n_embed_latent, eps=layernorm_eps, elementwise_affine=False),
        )

    def _init_weights(self, module):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        if isinstance(self.pos_embed, nn.Parameter):
            pos_embed = get_1d_sincos_pos_embed(self.n_embed, self.latent_dim)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca_layer(x)
        if isinstance(self.pos_embed, nn.Parameter):
            x = x + self.pos_embed
        for layer in self.encoder_layers:
            x = layer(x)
        h = self.encoder_latent_input(x)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_embed: int,
        n_embed_latent: int,
        n_head: int,
        n_head_cross: int,
        n_layer: int,
        n_inducing_points: int,
        dropout: float,
        bias: bool,
        multiple_of: int,
        layernorm_eps: float,
        norm_layer: str,
        shared_embedding: bool,
        use_adaln: bool = False,
    ):
        super().__init__()

        self.gene_embedding = nn.Embedding(n_genes + 1, n_embed) if not shared_embedding else nn.Identity()
        self.decoder_layers = nn.ModuleList([])

        self.decoder_latent_input = nn.Sequential(
            nn.LayerNorm(n_embed_latent, eps=layernorm_eps, elementwise_affine=False),
            nn.Linear(n_embed_latent, n_embed, bias=bias),
        )

        for _ in range(n_layer):
            self.decoder_layers.append(
                Block(
                    n_embed=n_embed,
                    n_head=n_head,
                    dropout=dropout,
                    bias=bias,
                    norm_layer=norm_layer,
                    multiple_of=multiple_of,
                    layernorm_eps=layernorm_eps,
                    use_adaln=use_adaln,
                )
            )
        self.decoder_cross_attention = CrossAttentionBlock(
            n_embed=n_embed,
            n_inducing_points=0,
            n_head=n_head_cross,
            dropout=dropout,
            bias=bias,
            norm_layer=norm_layer,
            multiple_of=multiple_of,
            layernorm_eps=layernorm_eps,
            use_adaln=use_adaln,
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self, x: torch.Tensor, genes: torch.Tensor, condition: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        x = self.decoder_latent_input(x)
        for layer in self.decoder_layers:
            x = layer(x, condition)
        q = self.gene_embedding(genes)
        output = self.decoder_cross_attention(x, q, condition)
        return output


####################
#        DiT       #
####################


class DiT(nn.Module):
    """Diffusion Transformer."""

    def __init__(
        self,
        n_embed: int,
        n_embed_input: int,
        n_layer: int,
        n_head: int,
        seq_len: int,
        dropout: float,
        bias: bool,
        norm_layer: str,
        multiple_of: int,
        layernorm_eps: float,
        class_vocab_sizes: dict[str, int],
        cfg_dropout_prob: float = 0.1,
        condition_strategy: Literal["mutually_exclusive", "joint"] = "mutually_exclusive",
    ):
        super().__init__()
        self.class_vocab_sizes = class_vocab_sizes
        self.cfg_dropout_prob = cfg_dropout_prob
        self.condition_strategy = condition_strategy
        self.class_embeddings = nn.ModuleDict()
        for class_name, vocab_size in class_vocab_sizes.items():
            use_cfg_embedding = int(cfg_dropout_prob > 0)
            self.class_embeddings[class_name] = nn.Embedding(vocab_size + use_cfg_embedding, n_embed)

        self.t_embedder = TimestepEmbedder(n_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, n_embed), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embed=n_embed,
                    n_head=n_head,
                    dropout=dropout,
                    bias=bias,
                    norm_layer=norm_layer,
                    multiple_of=multiple_of,
                    layernorm_eps=layernorm_eps,
                    use_adaln=True,  # this is only true, cause we have time
                    elementwise_affine=False,
                )
                for _ in range(n_layer)
            ]
        )

        self.n_embed = n_embed
        self.seq_len = seq_len

        self.input_proj = nn.Linear(n_embed_input, n_embed, bias=bias)
        self.final_layer = FinalLayerDit(n_embed, n_embed_input, bias, layernorm_eps)

        # Initialize weights (timestep MLP, pos_embed, class embeddings, adaLN layers, output layer)
        self.initialize_weights()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor],
        force_drop_ids: bool = True,
    ) -> torch.Tensor:
        t_embedding = self.t_embedder(t).unsqueeze(1)

        condition_embedding: torch.Tensor = self._get_condition_embedding(condition, force_drop_ids)

        if condition_embedding is not None:
            t_embedding = t_embedding + condition_embedding

        x = self.input_proj(x)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x=x, condition=t_embedding)

        x = self.final_layer(x, t_embedding)
        return x

    def forward_with_cfg_joint(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor] | None = None,
        cfg_scale: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Classifier-free guidance with additive conditioning strategy.

        During sampling:
        1. Get unconditional prediction (all classes = null tokens)
        2. For each condition class, add its guidance independently

        This matches the training strategy where only one class is active at a time.
        """
        batch_size = len(x)

        # Create unconditional condition (all null tokens) across all classes
        uncond_condition = {}
        for class_name in self.class_vocab_sizes.keys():
            null_token = self.class_vocab_sizes[class_name]
            uncond_condition[class_name] = torch.full((batch_size,), null_token, device=x.device, dtype=torch.long)

        # Get unconditional prediction (all null tokens)
        uncond_out = self.forward(x, t, uncond_condition, force_drop_ids=False)
        guided_out = uncond_out.clone()

        # Apply guidance for each condition class independently
        if condition is not None and cfg_scale is not None:
            cond_out = self.forward(x, t, condition, force_drop_ids=False)
            guided_out += (
                cfg_scale["cell_line"] * (cond_out - uncond_out)
                # cfg_scale["cell_type"] * (cond_out - uncond_out)
            )  # TODO since we are joining the cell type and cytokine, we need only one scale term, for readability we should change it later to a better name

        return guided_out

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor] | None = None,
        cfg_scale: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Efficient CFG sampling: first half unconditional, second half conditional."""
        batch_size = x.shape[0]
        len_half = batch_size // 2

        # Create unconditional condition (all null tokens) across all classes
        uncond_condition = {}
        for class_name in self.class_vocab_sizes.keys():
            null_token = self.class_vocab_sizes[class_name]
            uncond_condition[class_name] = torch.full((batch_size,), null_token, device=x.device, dtype=torch.long)

        uncond_out = self.forward(x, t, uncond_condition, force_drop_ids=False)

        # Split references
        uncond_out_half = uncond_out[:len_half]
        _cond_out_half = uncond_out[len_half:]
        cond_out_half = _cond_out_half.clone()

        # If no condition or scales, this leaves cond_out_half == base_half
        if condition is not None and cfg_scale is not None:
            x_half, t_half = x[len_half:], t[len_half:]

            for class_name, scale in cfg_scale.items():
                # Build per-class condition dict for the second half only
                single_condition_half = {class_name: condition[class_name][len_half:]}
                cond_pred_half = self.forward(x_half, t_half, single_condition_half, force_drop_ids=False)
                cond_out_half += scale * (cond_pred_half - _cond_out_half)

        return torch.cat([uncond_out_half, cond_out_half], dim=0)

    def _get_condition_embedding(self, condition: dict[str, torch.Tensor], force_drop_ids: bool = True) -> torch.Tensor:
        batch_size = next(iter(condition.values())).shape[0]
        device = next(iter(condition.values())).device

        if self.condition_strategy == "joint":
            return self._get_joint_condition_embedding(condition, batch_size, device)
        else:  # Default to "mutually_exclusive"
            return self._get_mutually_exclusive_condition_embedding(condition, batch_size, device, force_drop_ids)

    def _get_mutually_exclusive_condition_embedding(
        self, condition: dict[str, torch.Tensor], batch_size: int, device: torch.device, force_drop_ids: bool = True
    ) -> torch.Tensor:
        """During training: randomly select one condition class per batch and apply CFG dropout."""
        available_classes = [name for name in sorted(self.class_vocab_sizes.keys()) if name in condition]

        selected_class_idx = torch.randint(0, len(available_classes), (), device=device)

        # Optional per-sample dropout mask (device)
        if not self.training:
            assert not force_drop_ids, "force_drop_ids must be False when not training"

        if force_drop_ids:
            drop_mask = torch.rand(batch_size, device=device) < self.cfg_dropout_prob
        embeddings = []
        class_names = sorted(self.class_vocab_sizes.keys())
        for class_name in class_names:
            null_token = self.class_vocab_sizes[class_name]
            if class_name in available_classes:
                # position of class_name in available_classes (python-level only; constant across graph)
                i = available_classes.index(class_name)
                # broadcast over batch
                is_selected_b = (selected_class_idx == i).expand(batch_size)

                cond_vals = condition[class_name]
                null_vals = torch.full_like(cond_vals, null_token)
                if force_drop_ids:
                    cond_or_null = torch.where(drop_mask, null_vals, cond_vals)
                    final_vals = torch.where(is_selected_b, cond_or_null, null_vals)
                else:
                    final_vals = torch.where(is_selected_b, cond_vals, null_vals)
            else:
                final_vals = torch.full((batch_size,), null_token, device=device, dtype=torch.long)

            class_emb = self.class_embeddings[class_name](final_vals)
            embeddings.append(class_emb)

        return sum(embeddings).unsqueeze(1)

    def _get_joint_condition_embedding(
        self, condition: dict[str, torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """During training: use all available conditions jointly with independent CFG dropout."""
        available_classes = [name for name in sorted(self.class_vocab_sizes.keys()) if name in condition]
        if not available_classes:
            return torch.zeros(batch_size, 1, self.n_embed, device=device)

        # Apply CFG dropout independently to each condition class
        embeddings = []
        class_names = sorted(self.class_vocab_sizes.keys())

        if self.training:
            drop_mask = (
                torch.rand(batch_size, device=device) < self.cfg_dropout_prob
            )  # fixed masking for both types cells and cytokines
        else:
            drop_mask = torch.ones(batch_size, device=device) < 0  # no masking during inference

        for class_name in class_names:
            # Use the condition values with independent CFG dropout
            condition_values = condition[class_name]
            null_token = self.class_vocab_sizes[class_name]
            final_condition_values = torch.where(drop_mask, null_token, condition_values)
            class_emb = self.class_embeddings[class_name](final_condition_values)

            embeddings.append(class_emb)

        return sum(embeddings).unsqueeze(1)

    def initialize_weights(self):
        # Initialize transformer layers with xavier uniform
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.n_embed, self.seq_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize label embedding tables
        for embedding in self.class_embeddings.values():
            nn.init.normal_(embedding.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaln_modulation[-1].weight, 0)
            if block.adaln_modulation[-1].bias is not None:
                nn.init.constant_(block.adaln_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaln_modulation[-1].weight, 0)
        if self.final_layer.adaln_modulation[-1].bias is not None:
            nn.init.constant_(self.final_layer.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        if self.final_layer.linear.bias is not None:
            nn.init.constant_(self.final_layer.linear.bias, 0)
