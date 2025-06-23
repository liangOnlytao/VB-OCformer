import math
import torch

from torch import nn
from torch import Tensor

from ..bayesian_module import BayesLinear


class Query_Embedding(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 prior_pi: float = 1.0,
                 prior_mu1: float = 0.0,
                 prior_mu2: float = 0.0,
                 prior_sigma1: float = 0.1,
                 prior_sigma2: float = 0.4,
                 mu_init: float = 0.0,
                 sigma_init: float = -7.0,
                 bias: bool = True,
                 frozen: bool = False) -> None:
        super(Query_Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = BayesLinear(in_features = 1,
                                     out_features = d_model,
                                     prior_pi = prior_pi,
                                     prior_mu1 = prior_mu1,
                                     prior_mu2 = prior_mu2,
                                     prior_sigma1 = prior_sigma1,
                                     prior_sigma2 = prior_sigma2,
                                     mu_init = mu_init,
                                     sigma_init = sigma_init,
                                     bias = bias,
                                     frozen = frozen)
    
    def forward(self,
                x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)

        return x


class Query_PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 max_length: int = 5000) -> None:
        super(Query_PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, d_model)
        pos = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self,
                x: Tensor) -> Tensor:
        return self.pe[:, :x.size(1)].requires_grad_(False)


class Query_BayesformerEmbedding(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 dropout_rate: float = 0.1,
                 max_length: int = 5000,
                 prior_pi: float = 1.0,
                 prior_mu1: float = 0.0,
                 prior_mu2: float = 0.0,
                 prior_sigma1: float = 0.1,
                 prior_sigma2: float = 0.4,
                 mu_init: float = 0.0,
                 sigma_init: float = -7.0,
                 bias: bool = True,
                 frozen: bool = False) -> None:
        super(Query_BayesformerEmbedding, self).__init__()
        self.embedding = Query_Embedding(d_model = d_model,
                                         prior_pi = prior_pi,
                                         prior_mu1 = prior_mu1,
                                         prior_mu2 = prior_mu2,
                                         prior_sigma1 = prior_sigma1,
                                         prior_sigma2 = prior_sigma2,
                                         mu_init = mu_init,
                                         sigma_init = sigma_init,
                                         bias = bias,
                                         frozen = frozen)
        self.positionalencoding = Query_PositionalEncoding(d_model=d_model, max_length=max_length)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self,
                x: Tensor) -> Tensor:
        x_embedding = self.embedding(x)
        x_positionalencoding = self.positionalencoding(x)
        x = self.dropout(x_embedding + x_positionalencoding)
        
        return x


class Key_Value_Embedding(nn.Module):
    def __init__(self,
                 seq_len: int,
                 d_model: int = 512,
                 prior_pi: float = 1.0,
                 prior_mu1: float = 0.0,
                 prior_mu2: float = 0.0,
                 prior_sigma1: float = 0.1,
                 prior_sigma2: float = 0.4,
                 mu_init: float = 0.0,
                 sigma_init: float = -7.0,
                 bias: bool = True,
                 frozen: bool = False) -> None:
        super(Key_Value_Embedding, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.embedding = BayesLinear(in_features = seq_len,
                                     out_features = seq_len * d_model,
                                     prior_pi = prior_pi,
                                     prior_mu1 = prior_mu1,
                                     prior_mu2 = prior_mu2,
                                     prior_sigma1 = prior_sigma1,
                                     prior_sigma2 = prior_sigma2,
                                     mu_init = mu_init,
                                     sigma_init = sigma_init,
                                     bias = bias,
                                     frozen = frozen)
    
    def forward(self,
                x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = x.view(-1, self.seq_len, self.d_model)
        x = x * math.sqrt(self.d_model)

        return x


class Key_Value_PositionalEncoding(nn.Module):
    def __init__(self,
                 seq_len: int,
                 d_model: int = 512) -> None:
        super(Key_Value_PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self) -> Tensor:
        return self.pe.requires_grad_(False)


class Key_Value_BayesformerEmbedding(nn.Module):
    def __init__(self,
                 seq_len: int,
                 d_model: int = 512,
                 dropout_rate: float = 0.1,
                 prior_pi: float = 1.0,
                 prior_mu1: float = 0.0,
                 prior_mu2: float = 0.0,
                 prior_sigma1: float = 0.1,
                 prior_sigma2: float = 0.4,
                 mu_init: float = 0.0,
                 sigma_init: float = -7.0,
                 bias: bool = True,
                 frozen: bool = False) -> None:
        super(Key_Value_BayesformerEmbedding, self).__init__()
        self.embedding = Key_Value_Embedding(seq_len = seq_len,
                                             d_model = d_model,
                                             prior_pi = prior_pi,
                                             prior_mu1 = prior_mu1,
                                             prior_mu2 = prior_mu2,
                                             prior_sigma1 = prior_sigma1,
                                             prior_sigma2 = prior_sigma2,
                                             mu_init = mu_init,
                                             sigma_init = sigma_init,
                                             bias = bias,
                                             frozen = frozen)
        self.positionalencoding = Key_Value_PositionalEncoding(seq_len=seq_len, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self,
                x: Tensor) -> Tensor:
        x_embedding = self.embedding(x)
        x_positionalencoding = self.positionalencoding()
        x = self.dropout(x_embedding + x_positionalencoding)

        return x