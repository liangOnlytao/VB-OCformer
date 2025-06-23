import torch

from torch import nn
from torch import Tensor

from ..bayesian_module import BayesModule
from ..bayesian_module import bayesian_modules

from typing import Union


class KLDiv(nn.Module):
    def __init__(self,
                 model: Union[nn.Module, BayesModule]) -> None:
        super(KLDiv, self).__init__()
        self.model = model

    def forward(self) -> Tensor:
        kl_divergence = torch.zeros(1)
        count = 0

        for module in self.model.modules():
            if isinstance(module, bayesian_modules):
                kl_divergence = kl_divergence.to(device=module.lvposterior.device)
                kl_divergence += module.lvposterior - module.lprior
                count += len(module.weight_mu.view(-1))
                if module.bias:
                    count += len(module.bias_mu.view(-1))

        return kl_divergence / count


class ELBOLoss(nn.Module):
    def __init__(self,
                 model: Union[nn.Module, BayesModule],
                 inner_loss: nn.Module,
                 kl_weight: float,
                 num_samples: int) -> None:
        super(ELBOLoss, self).__init__()
        self.set_model(model)

        self.inner_loss = inner_loss
        self.kl_weight = kl_weight
        self.num_samples = num_samples
    
    def forward(self,
                x: Tensor,
                y: Tensor) -> Tensor:
        aggregated_elbo = torch.zeros(1, device=x.device)

        for _ in range(self.num_samples):
            y_hat = self.model(x)
            aggregated_elbo += self.inner_loss(y_hat, y)
            aggregated_elbo += self.kl_weight * self._kl_div().to(device=x.device)
        
        return aggregated_elbo / self.num_samples
    
    def set_model(self,
                  model: Union[nn.Module, BayesModule]) -> None:
        self.model = model
        self._kl_div = KLDiv(model)