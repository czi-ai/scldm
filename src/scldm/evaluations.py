import math
from functools import partial
from typing import Literal

import ot
import torch
from torch import nn


class RBFKernel(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()

        self.scale = scale

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = (x**2).sum(dim=1, keepdim=True)  # Bx x 1
        y_norm = (y**2).sum(dim=1, keepdim=True)  # By x 1
        squared_ell_2 = x_norm - 2 * x @ y.T + y_norm.T  # Bx x By

        return torch.exp(-self.scale * squared_ell_2)


class BrayCurtisKernel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Bx x 1 x D
        y = y.unsqueeze(0)  # 1 x By x D

        numerator = torch.abs(x - y).sum(dim=2)  # Bx x By
        denominator = torch.abs(x + y).sum(dim=2) + 1e-8

        return 1 - numerator / denominator


class TanimotoKernel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Bx x 1 x D
        y = y.unsqueeze(0)  # 1 x By x D

        numerator = (x * y).sum(dim=2)  # Bx x By
        denominator = (x + y - x * y).sum(dim=2) + 1e-8

        return numerator / denominator


class RuzickaKernel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Bx x 1 x D
        y = y.unsqueeze(0)  # 1 x By x D

        numerator = torch.min(x, y).sum(dim=2)  # Bx x By
        denominator = torch.max(x, y).sum(dim=2) + 1e-8

        return numerator / denominator


class MMDLoss(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_xx = self.kernel(x, x)
        k_yy = self.kernel(y, y)
        k_xy = self.kernel(x, y)

        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Literal["emd", "sinkhorn"] = "emd",
    reg: float = 0.05,
    power: int = 2,
) -> float:
    assert power == 1 or power == 2

    if method == "emd" or method is None:
        ot_fn = ot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(ot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = ot.unif(x0.shape[0], type_as=x0), ot.unif(x1.shape[0], type_as=x1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M, numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)
    return ret
