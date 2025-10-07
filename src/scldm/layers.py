"""
Adapted from:

- https://github.com/epfml/llm-baselines/blob/main/src/models
- https://github.com/pytorch-labs/attention-gym/blob/main/examples
"""

import math
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

SCORE_MOD = {
    "noop": None,
}


NORM_LAYERS = {
    "layernorm": nn.LayerNorm,
}


def log1p_transform(genes: torch.Tensor, counts: torch.Tensor, zero_encoding: bool = False) -> torch.Tensor:
    if zero_encoding:
        return genes * torch.where(counts == 0, torch.tensor(-1.0, device=counts.device), torch.log1p(counts))
    return genes * torch.log1p(counts)


PROJ_FUNC = {
    "log1p": log1p_transform,
    "log1pzero": partial(log1p_transform, zero_encoding=True),
}


def modulate(x, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Applies adaptive layer normalization modulation."""
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift


class InputTransformerVAE(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_embed: int,
        agg_func: Literal["log1p", "anscombe", "sqrt", "proj", "projconcat", "softbin", "log1pzero"],
    ):
        super().__init__()
        self.gene_embedding = nn.Embedding(n_genes + 1, n_embed)

        self.projection: nn.Module | Callable[[Any], Any] = PROJ_FUNC[agg_func]
        if agg_func in ["proj", "projconcat", "softbin"]:
            self.projection = PROJ_FUNC[agg_func](n_embed)

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
    ) -> torch.Tensor:
        genes_emb = self.gene_embedding(genes)
        output = self.projection(genes_emb, counts.unsqueeze(-1))
        return output


class SelfAttention(nn.Module):
    def __init__(
        self,
        n_embed: int,
        n_head: int,
        dropout: float,
        bias: bool,
    ):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.n_embed = n_embed
        self.dropout = dropout

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embed, 3 * self.n_embed, bias=bias)
        # output projection
        self.c_proj = nn.Linear(self.n_embed, self.n_embed, bias=bias)
        # regularization
        self.resid_dropout = nn.Dropout(self.dropout)
        self.flex_attention = flex_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        k = k.view(B, S, self.n_head, D // self.n_head)
        q = q.view(B, S, self.n_head, D // self.n_head)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = v.view(B, S, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nH, S, hD)

        y: torch.Tensor = self.flex_attention(q, k, v, block_mask=None, score_mod=None, return_lse=False)
        y = y.transpose(1, 2).contiguous().view(B, S, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embed: int, multiple_of: int):
        super().__init__()

        hidden_dim = n_embed * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, n_embed, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(nn.functional.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(
        self,
        n_embed: int,
        n_head: int,
        dropout: float,
        bias: bool,
        norm_layer: str,
        multiple_of: int,
        layernorm_eps: float,
        use_adaln: bool = False,
        elementwise_affine: bool = True,
    ):
        super().__init__()

        self.ln_1 = NORM_LAYERS[norm_layer](n_embed, eps=layernorm_eps, elementwise_affine=elementwise_affine)
        self.ln_2 = NORM_LAYERS[norm_layer](n_embed, eps=layernorm_eps, elementwise_affine=elementwise_affine)

        self.attn = SelfAttention(
            n_embed=n_embed,
            n_head=n_head,
            dropout=dropout,
            bias=bias,
        )

        self.mlp = MLP(n_embed=n_embed, multiple_of=multiple_of)

        self.use_adaln = use_adaln
        if use_adaln:
            self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(n_embed, 6 * n_embed, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_adaln:
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(condition).chunk(
                6, dim=-1
            )
            norm1_out = modulate(self.ln_1(x), scale_attn, shift_attn)
            x = x + gate_attn * self.attn(norm1_out)
            norm2_out = modulate(self.ln_2(x), scale_mlp, shift_mlp)
            x_ = gate_mlp * self.mlp(norm2_out)
            x = x + x_
        else:
            x = x + self.attn(self.ln_1(x))
            x_ = self.mlp(self.ln_2(x))
            x = x + x_
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        n_embed: int,
        n_head: int,
        dropout: float,
        bias: bool,
    ):
        super().__init__()

        self.n_head = n_head
        self.n_embed = n_embed

        self.c_attn = nn.Linear(n_embed, 2 * n_embed, bias=bias)  # key, value projections for x
        self.c_attn_q = nn.Linear(n_embed, n_embed, bias=bias)  # key, projection for q
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)  # output projection
        self.flex_attention = flex_attention
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        _, M, Dout = q.shape  # get seq_len of inducing points

        k, v = self.c_attn(x).split(self.n_embed, dim=-1)
        q = self.c_attn_q(q)

        k = k.view(B, S, self.n_head, Dout // self.n_head)
        v = v.view(B, S, self.n_head, Dout // self.n_head)
        q = q.view(B, M, self.n_head, Dout // self.n_head)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, nH, S, hD)

        y: torch.Tensor = self.flex_attention(q, k, v, block_mask=None, score_mod=None, return_lse=False)
        y = y.transpose(1, 2).contiguous().view(B, M, Dout)
        pooled_x = self.resid_dropout(self.c_proj(y))  # B, M, Dout

        return pooled_x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        n_embed: int,
        n_inducing_points: int,
        n_head: int,
        dropout: float,
        bias: bool,
        norm_layer: str,
        multiple_of: int,
        layernorm_eps: float,
        use_adaln: bool = False,
    ):
        super().__init__()

        self.inducing_points = (
            None
            if n_inducing_points == 0
            else nn.Parameter(torch.randn(n_inducing_points, n_embed), requires_grad=True)
        )

        self.ln_1 = NORM_LAYERS[norm_layer](n_embed, eps=layernorm_eps)
        self.ln_1q = NORM_LAYERS[norm_layer](n_embed, eps=layernorm_eps)

        self.attn = CrossAttention(
            n_embed=n_embed,
            n_head=n_head,
            dropout=dropout,
            bias=bias,
        )

        self.ln_2 = NORM_LAYERS[norm_layer](n_embed, eps=layernorm_eps)
        self.mlp = MLP(n_embed=n_embed, multiple_of=multiple_of)
        self.use_adaln = use_adaln
        if use_adaln:
            self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(n_embed, 6 * n_embed, bias=True))
            self.adaln_modulation_q = nn.Sequential(nn.SiLU(), nn.Linear(n_embed, 2 * n_embed, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, _, _ = x.shape
        if self.inducing_points is not None and q is None:
            q = self.inducing_points.expand(B, -1, -1)  # expand inducing points over batches without allocation
        if self.use_adaln:
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(condition).chunk(
                6, dim=-1
            )
            shift_q, scale_q = self.adaln_modulation_q(condition).chunk(2, dim=-1)
            norm1_xout = modulate(self.ln_1(x), scale_attn, shift_attn)
            norm1_qout = modulate(self.ln_1q(q), scale_q, shift_q)
            x = q + gate_attn * self.attn(norm1_xout, norm1_qout)
            norm2_out = modulate(self.ln_2(x), scale_mlp, shift_mlp)
            x_ = gate_mlp * self.mlp(norm2_out)
            x = x + x_
        else:
            attn_output = self.attn(self.ln_1(x), self.ln_1q(q))
            x = q + attn_output
            x_ = self.mlp(self.ln_2(x))
            x = x + x_
        return x

    def extra_repr(self):
        return f"(inducing_points): Parameter(shape={self.inducing_points.shape})"


##########################
#     Layers for DiT     #
##########################
class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Creates sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


def get_1d_sincos_pos_embed(embed_dim, seq_len):
    assert embed_dim % 2 == 0, "Embedding dimension must be even"
    positions = np.arange(seq_len, dtype=np.float32)

    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)  # (embed_dim // 2,)

    pos = positions.reshape(-1, 1)  # (seq_len, 1)
    omega = omega.reshape(1, -1)  # (1, embed_dim // 2)

    out = pos * omega  # (seq_len, embed_dim // 2)

    emb_sin = np.sin(out)  # (seq_len, embed_dim // 2)
    emb_cos = np.cos(out)  # (seq_len, embed_dim // 2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (seq_len, embed_dim)

    return emb


class FinalLayerDit(nn.Module):
    """The final layer of DiT."""

    def __init__(self, n_embed: int, n_embed_input: int, bias: bool, layernorm_eps: float):
        super().__init__()
        self.norm_final = nn.LayerNorm(n_embed, elementwise_affine=False, eps=layernorm_eps)
        self.linear = nn.Linear(n_embed, n_embed_input, bias=bias)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(n_embed, 2 * n_embed, bias=bias))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
