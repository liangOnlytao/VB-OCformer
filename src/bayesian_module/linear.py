import math
import torch

from torch import nn
from torch import Tensor
from torch.nn import init
from torch.nn import functional as F

from .sampler import TrainableDistribution, CenteredGaussianMixture


class BayesLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 prior_pi: float = 1.0,
                 prior_mu1: float = 0.0,
                 prior_mu2: float = 0.0,
                 prior_sigma1: float = 0.1,
                 prior_sigma2: float = 0.4,
                 mu_init: float = 0.0,
                 sigma_init: float = -7.0,
                 bias: bool = True,
                 frozen: bool = False) -> None:
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.prior_pi = prior_pi
        self.prior_mu1 = prior_mu1
        self.prior_mu2 = prior_mu2
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2

        self.mu_init = mu_init
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = True
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = False
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
        
        self.frozen = frozen

        self.reset_parameters()

        self.weight_sampler = TrainableDistribution(self.weight_mu, self.weight_sigma)
        if bias:
            self.bias_sampler = TrainableDistribution(self.bias_mu, self.bias_sigma)
        
        self.weight_prior_dist = CenteredGaussianMixture(prior_pi, prior_mu1, prior_mu2, prior_sigma1, prior_sigma2)
        if bias:
            self.bias_prior_dist = CenteredGaussianMixture(prior_pi, prior_mu1, prior_mu2, prior_sigma1, prior_sigma2)
    
    def reset_parameters(self) -> None:
        init.normal_(self.weight_mu, mean=self.mu_init, std=0.1)
        init.normal_(self.weight_sigma, mean=self.sigma_init, std=0.1)

        if self.bias_mu is not None:
            init.normal_(self.bias_mu, mean=self.mu_init, std=0.1)
            init.normal_(self.bias_sigma, mean=self.sigma_init, std=0.1)
    
    def freeze(self) -> None:
        self.frozen = True
    
    def unfreeze(self) -> None:
        self.frozen = False
    
    def forward(self,
                x: Tensor) -> Tensor:
        if self.frozen:
            return F.linear(x, self.weight_mu, self.bias_mu)
        else:
            weight = self.weight_sampler.sample()

            if self.bias:
                bias = self.bias_sampler.sample()
                bias_lposterior = self.bias_sampler.log_posterior()
                bias_lprior = self.bias_prior_dist.log_prob(bias)
            else:
                bias, bias_lposterior, bias_lprior = None, 0, 0
            
            self.lvposterior = self.weight_sampler.log_posterior() + bias_lposterior
            self.lprior = self.weight_prior_dist.log_prob(weight) + bias_lprior
            
            return F.linear(x, weight, bias)
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias}"