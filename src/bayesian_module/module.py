import torch

from torch import nn
from torch import Tensor

from .bayesian_modules import bayesian_modules


class BayesModule(nn.Module):
    def __init__(self) -> None:
        super(BayesModule, self).__init__()
    
    def freeze(self):
        for module in self.modules():
            if isinstance(module, bayesian_modules):
                module.freeze()
    
    def unfreeze(self):
        for module in self.modules():
            if isinstance(module, bayesian_modules):
                module.unfreeze()