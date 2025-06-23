import copy
import math
import torch

from torch import nn
from torch import Tensor
from torch.nn import functional as F

from typing import Callable, Optional


class MultiheadAttention(nn.Module):
    def __init__(self,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 bias: bool = True) -> None:
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(in_features=self.embed_dim, out_features=self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(in_features=self.embed_dim, out_features=self.num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(in_features=self.embed_dim, out_features=self.num_heads * self.head_dim, bias=bias)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.out_proj = nn.Linear(in_features=self.num_heads * self.head_dim, out_features=self.embed_dim, bias=bias)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor) -> Tensor:
        batch_size, _, _ = query.size()
        
        q = self.q_proj(query).contiguous().view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).contiguous().view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).transpose(2, 3)
        v = self.v_proj(value).contiguous().view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output_weights = torch.matmul(q, k) / math.sqrt(self.head_dim)
        attn_output_weights = attn_output_weights.softmax(dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = torch.matmul(attn_output_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 num_heads: int = 8,
                 dim_feedforward: int = 2048,
                 dropout_rate: float = 0.1,
                 activation: Callable = F.relu,
                 layer_norm_eps: float = 1e-5,
                 bias: bool = True,
                 norm_first: bool = False) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.multihead_attn = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout_rate=dropout_rate, bias=bias)
        
        self.linear1 = nn.Linear(in_features=d_model, out_features=dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear2 = nn.Linear(in_features=dim_feedforward, out_features=d_model, bias=bias)

        self.norm1 = nn.LayerNorm(normalized_shape=d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.activation = activation
        self.norm_first = norm_first
    
    def forward(self,
                src: Tensor,
                memory: Tensor) -> Tensor:
        output = src
        if self.norm_first:
            output = output + self._mha_block(self.norm1(output), self.norm1(memory))
            output = output + self._ff_block(self.norm2(output))
        else:
            output = self.norm1(output + self._mha_block(output, memory))
            output = self.norm2(output + self._ff_block(output))

        return output
    
    def _mha_block(self,
                   x: Tensor,
                   mem: Tensor) -> Tensor:
        x = self.multihead_attn(x, mem, mem)
        return self.dropout1(x)
    
    def _ff_block(self,
                  x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(self,
                 query_encoder_embedding,
                 key_value_embedding,
                 encoder_layer,
                 num_layers: int = 6,
                 norm: Optional[nn.Module] = None) -> None:
        super(TransformerEncoder, self).__init__()
        self.query_encoder_embedding = query_encoder_embedding
        self.key_value_embedding = key_value_embedding
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm
    
    def forward(self,
                src: Tensor) -> Tensor:
        output = self.query_encoder_embedding(src)
        mem = self.key_value_embedding(src)
        for mod in self.layers:
            output = mod(output, mem)
        if self.norm is not None:
            output = self.norm(output)
        
        return output