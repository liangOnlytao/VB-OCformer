import torch
import numpy as np

from torch import nn
from torch import Tensor
from torch import distributions

from typing import Optional


class TrainableDistribution(nn.Module):
    lsqrt2pi = torch.tensor(np.log(np.sqrt(2 * np.pi)))

    def __init__(self,
                 mu: Tensor,
                 rho: Tensor) -> None:
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.sigma = None
        self.weight = None
    
    def sample(self) -> Tensor:
        weight_eps = torch.randn(size=self.mu.shape, device=self.mu.device)
        self.sigma = torch.log1p(torch.exp(self.rho)).to(device=self.mu.device)
        self.weight = self.mu + self.sigma * weight_eps
        return self.weight
    
    def log_posterior(self,
                      weight: Optional[Tensor] = None) -> Tensor:
        assert self.weight is not None and self.sigma is not None
        if weight is None:
            weight = self.weight
        lposterior = self.lsqrt2pi + torch.log(self.sigma) + (((weight - self.mu) ** 2) / (2 * self.sigma**2))
        return -lposterior.sum()


class CenteredGaussianMixture(nn.Module):
    def __init__(self,
                 pi: float,
                 mu1: float,
                 mu2: float,
                 sigma_1: float,
                 sigma_2: float) -> None:
        super(CenteredGaussianMixture, self).__init__()
        self.register_buffer("pi", torch.tensor([pi, 1 - pi]))
        self.register_buffer("mus", torch.tensor([mu1, mu2]))
        self.register_buffer("sigmas", torch.tensor([sigma_1, sigma_2]))
    
    def log_prob(self,
                 weight: Tensor) -> Tensor:
        mix = distributions.Categorical(self.pi)
        normals = distributions.Normal(self.mus, self.sigmas)
        distribution = distributions.MixtureSameFamily(mix, normals)
        return distribution.log_prob(weight).sum()