import torch

from torch import nn
from torch import Tensor
from torch.nn import functional as F

from .embedding import Query_BayesformerEmbedding, Key_Value_BayesformerEmbedding
from .encoder import TransformerEncoder
from .encoder import TransformerEncoderLayer

from typing import Callable


class Bayesformer(nn.Module):
    def __init__(self,
                 seq_len: int,
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
        super(Bayesformer, self).__init__()
        query_encoder_embedding = Query_BayesformerEmbedding(d_model = d_model,
                                                             dropout_rate = dropout_rate,
                                                             max_length = max_length,
                                                             prior_pi = prior_pi,
                                                             prior_mu1 = prior_mu1,
                                                             prior_mu2 = prior_mu2,
                                                             prior_sigma1 = prior_sigma1,
                                                             prior_sigma2 = prior_sigma2,
                                                             mu_init = mu_init,
                                                             sigma_init = sigma_init,
                                                             bias = bias,
                                                             frozen = frozen)
        key_value_embedding = Key_Value_BayesformerEmbedding(seq_len = seq_len,
                                                             d_model = d_model,
                                                             dropout_rate = dropout_rate,
                                                             prior_pi = prior_pi,
                                                             prior_mu1 = prior_mu1,
                                                             prior_mu2 = prior_mu2,
                                                             prior_sigma1 = prior_sigma1,
                                                             prior_sigma2 = prior_sigma2,
                                                             mu_init = mu_init,
                                                             sigma_init = sigma_init,
                                                             bias = bias,
                                                             frozen = frozen)
        encoder_layer = TransformerEncoderLayer(d_model = d_model,
                                                num_heads = num_heads,
                                                dim_feedforward = dim_feedforward,
                                                dropout_rate = dropout_rate,
                                                activation = activation,
                                                layer_norm_eps = layer_norm_eps,
                                                bias = bias,
                                                norm_first = norm_first)
        if encoder_norm:
            encoder_norm = nn.LayerNorm(normalized_shape=d_model, eps=layer_norm_eps, bias=bias)
        else:
            encoder_norm = None
        self.encoder = TransformerEncoder(query_encoder_embedding = query_encoder_embedding,
                                          key_value_embedding = key_value_embedding,
                                          encoder_layer = encoder_layer,
                                          num_layers = num_encoder_layers,
                                          norm = encoder_norm)

    def forward(self,
                src: Tensor) -> Tensor:
        output = self.encoder(src)

        return output