import torch

from torch import nn
from torch import Tensor
from torch.nn import functional as F

from ..module import Bayesformer
from ..bayesian_module import BayesModule

from typing import List, Callable


class VBOCformerNeuralNetwork(BayesModule):
    def __init__(self,
                 seq_len: int,
                 out_features: int,
                 hidden_layer_sizes: List[int],
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout_rate: float = 0.1,
                 max_length: int = 5000,
                 activation: Callable = F.relu,
                 layer_norm_eps: float = 1e-5,
                 encoder_norm: bool = True,
                 prior_pi: float = 1.0,
                 prior_mu1: float = 0.0,
                 prior_mu2: float = 0.0,
                 prior_sigma1: float = 0.1,
                 prior_sigma2: float = 0.4,
                 mu_init: float = 0.0,
                 sigma_init: float = -7.0,
                 bias: bool = True,
                 frozen: bool = False,
                 norm_first: bool = False) -> None:
        super(VBOCformerNeuralNetwork, self).__init__()
        self.transformer = Bayesformer(seq_len = seq_len,
                                       d_model = d_model,
                                       num_heads = num_heads,
                                       num_encoder_layers = num_encoder_layers,
                                       dim_feedforward = dim_feedforward,
                                       dropout_rate = dropout_rate,
                                       max_length = max_length,
                                       activation = activation,
                                       layer_norm_eps = layer_norm_eps,
                                       encoder_norm = encoder_norm,
                                       prior_pi = prior_pi,
                                       prior_mu1 = prior_mu1,
                                       prior_mu2 = prior_mu2,
                                       prior_sigma1 = prior_sigma1,
                                       prior_sigma2 = prior_sigma2,
                                       mu_init = mu_init,
                                       sigma_init = sigma_init,
                                       bias = bias,
                                       frozen = frozen,
                                       norm_first = norm_first)
        
        in_features = seq_len * d_model
        in_dims = [in_features] + hidden_layer_sizes
        out_dims = hidden_layer_sizes + [out_features]
        linears = nn.ModuleList()
        for dims in zip(in_dims, out_dims):
            linears.append(nn.Linear(in_features=dims[0], out_features=dims[1], bias=bias))
        self.linears = linears

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self,
                x: Tensor) -> Tensor:
        output = self.transformer(x).flatten(start_dim=1, end_dim=-1)
        output = self.activation(self.dropout(output))
        for i in range(len(self.linears) - 1):
            output = self.activation(self.dropout(self.linears[i](output)))
        output = self.linears[-1](output)

        return output.squeeze(-1)