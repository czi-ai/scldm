from typing import Any

import torch
import torch.distributions as dist
import torch.nn as nn


class BasePrior(nn.Module):
    """Base class for latent priors."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> Any:
        """Transform input to distribution parameters."""
        raise NotImplementedError

    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample from the prior distribution."""
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the log-probability."""
        raise NotImplementedError

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the loss (e.g., KL divergence)."""
        raise NotImplementedError


###########################
# STANDARD GAUSSIAN PRIOR #
###########################
class StandardPrior(BasePrior):
    def __init__(self, n_latent: int, n_embed: int = 1):
        super().__init__()

        self.n_latent = n_latent

        # Store parameters in a more elegant way
        if n_embed == 0:
            self.loc = nn.Parameter(torch.zeros(1, n_latent), requires_grad=False)
            self.scale = nn.Parameter(torch.ones(1, n_latent), requires_grad=False)
        else:
            self.loc = nn.Parameter(torch.zeros(1, n_latent, n_embed), requires_grad=False)
            self.scale = nn.Parameter(torch.ones(1, n_latent, n_embed), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.distributions.Normal:
        # x has shape (B, n_latent, embedding)
        # Parameters already have the right shape for broadcasting
        return dist.Normal(loc=self.loc, scale=self.scale)

    def sample(self, n_samples: int):
        # Create a standard normal distribution with the right shape
        return dist.Normal(self.loc, self.scale).sample((n_samples,)).squeeze(1)

    def log_prob(self, x: torch.Tensor):
        # x has shape (B, n_latent, embedding)
        return dist.Normal(self.loc, self.scale).log_prob(x)

    def loss(self, x: torch.Tensor):
        return self.log_prob(x)
