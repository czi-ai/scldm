import torch
import torch.distributions as dist
import torch.nn as nn
from scvi.distributions import NegativeBinomial as NegativeBinomialSCVI
from torch.distributions import Distribution


###########################
#     GAUSSIAN LAYERS     #
###########################
class GaussianLinearLayer(nn.Module):
    def __init__(
        self,
        n_hidden: int,
        n_latent: int,
    ):
        super().__init__()
        self.loc = nn.Linear(n_hidden, n_latent, bias=True)
        self.scale = nn.Linear(n_hidden, n_latent, bias=True)

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        # location
        loc = self.loc(x)
        # scale
        log_scale = self.scale(x)
        log_scale = nn.functional.hardtanh(log_scale, min_val=-7.0, max_val=5.0)
        scale = torch.exp(log_scale)
        return dist.Normal(loc, scale)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        distribution = self.forward(x)
        return distribution.rsample()

    def log_prob(self, x: torch.Tensor, loc: torch.Tensor | None, scale: torch.Tensor | None) -> torch.Tensor:
        if (loc is None) or (scale is None):
            distribution = self.forward(x)
        else:
            distribution = dist.Normal(loc, scale)
        log_p = distribution.log_prob(x)
        return log_p

    def loss(self, x: torch.Tensor, loc: torch.Tensor | None, scale: torch.Tensor | None) -> torch.Tensor:
        return self.log_prob(x, loc, scale)


############################
#     NBinomial LAYERS     #
############################
class NegativeBinomialTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,
        shared_theta: bool = False,
        n_embed: int | None = None,
        norm_layer: str = "layernorm",
        layernorm_eps: float = 1e-8,
        eps_: float = 1e-6,
        t: float = 1.0,
    ):
        super().__init__()
        self.shared_theta = shared_theta

        if shared_theta:
            self.theta = nn.Embedding(n_genes + 1, 1)
            torch.nn.init.ones_(self.theta.weight)
            self.params = nn.Linear(n_embed, 1, bias=True)
        else:
            self.theta = None
            self.params = nn.Linear(n_embed, 2, bias=True)

        self.eps_ = eps_
        self.t = t

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.theta, nn.Embedding):
            mu = self.params(counts)
            theta = self.theta(genes.long())
        else:
            params = self.params(counts)
            mu, theta = torch.chunk(params, 2, dim=-1)
        mu, theta = mu.squeeze(-1), torch.exp(theta).squeeze(-1)
        mu = nn.functional.softmax(mu / self.t, dim=1) * library_size
        return mu, theta

    def log_prob(self, counts: torch.Tensor, genes: torch.Tensor, total_counts: torch.Tensor) -> torch.Tensor:
        distribution = self.forward(counts, genes, total_counts)
        return distribution.log_prob(counts)


class NegativeBinomialLinearLayer(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,
        n_hidden: int,
        shared_theta: bool = False,
    ):
        super().__init__()
        self.shared_theta = shared_theta

        self.mu = nn.Linear(n_hidden, n_genes, bias=True)
        if self.shared_theta:
            self.theta: nn.Parameter | nn.Linear = nn.Parameter(torch.ones(n_genes), requires_grad=True)
        else:
            self.theta: nn.Linear = nn.Linear(n_hidden, n_genes, bias=True)
        self.softplus = nn.Softplus()

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
    ) -> Distribution:
        mu = self.mu(counts)
        if isinstance(self.theta, nn.Parameter):
            theta = self.softplus(self.theta)
        else:
            theta = self.softplus(self.theta(counts))
        mu = nn.functional.softmax(mu, dim=1)
        mu = mu * library_size
        return NegativeBinomialSCVI(mu=mu, theta=theta)

    def log_prob(self, x: torch.Tensor, total_counts: torch.Tensor) -> torch.Tensor:
        distribution = self.forward(x, total_counts)
        return distribution.log_prob(x)
