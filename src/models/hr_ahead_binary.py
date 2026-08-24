# legacy: pre-1.4 model, not wired to any experiment. Batch-style but still
# assumes the old square (B,T,T) mask; migrate onto models.protocols.Model
# with the plain (B,T) key-padding mask from data.datasets.types before reuse.

from typing import Protocol, NamedTuple
from utils import component
from pydantic import BaseModel, PositiveInt, PositiveFloat
import torch
from torch import Tensor
import torch.nn as nn

class ModelConfig(BaseModel):
    process_heads: int
    n_layers: PositiveInt
    vocab_size: PositiveInt
    embed_dim: PositiveInt
    hidden_dim: PositiveInt
    n_out: PositiveInt
    dropout: PositiveFloat

class LayerConfig(BaseModel):
    process_heads: PositiveInt
    embed_dim: PositiveInt
    hidden_dim: PositiveInt
    dropout: PositiveFloat


class Batch(Protocol):
    x: Tensor
    mask: Tensor


class Output(NamedTuple):
    logits: Tensor
    probs: Tensor

class MultiHeadSelfAttn(nn.Module):
    def __init__(
        self,
        dim_in: int,
        hidden_dim: int,
        dim_out: int,
        n_heads: int,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.QKV_proj = nn.Linear(dim_in, hidden_dim * n_heads * 3, device=device, dtype=dtype)
        self.head = nn.Linear(hidden_dim * n_heads, dim_out, device=device, dtype=dtype)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B, T, C = x.shape
        QKV = self.QKV_proj(x)
        Q, K, V = torch.chunk(QKV, 3, dim=-1)
        Q = Q.reshape(B, T, -1, self.n_heads).permute(0, 3, 1, 2)
        K = K.reshape(B, T, -1, self.n_heads).permute(0, 3, 1, 2)
        V = V.reshape(B, T, -1, self.n_heads).permute(0, 3, 1, 2)
        
        out = nn.functional.scaled_dot_product_attention(Q, K, V, mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1), dropout_p=self.dropout)
        
        out = out.transpose(1, 2).contiguous().flatten(2,-1)
        out = self.head(out)
        return out

class MLP(nn.Module):
    def __init__(
            self,
            config: LayerConfig,
            device: torch.device,
            dtype: torch.dtype,
    ):
        super().__init__() 
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.embed_dim, device=device, dtype=dtype),
            nn.GELU(),
            nn.Dropout(p=config.dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

class HAttnBlock(nn.Module):
    def __init__(self, 
                 config: LayerConfig,
                 device: torch.device,
                 dtype: torch.dtype,
                 ):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype

        self.process_attn = MultiHeadSelfAttn(config.embed_dim, config.hidden_dim, config.hidden_dim, config.process_heads, config.dropout, device, dtype)

        self.mlp = MLP(config, device, dtype) 
        
    def forward(self, x:Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        B, T, C = x.shape
        x = nn.functional.rms_norm(x, (x.size(-1), ))

        out = x + self.mlp(
            self.process_attn(x, mask)
        ).contiguous()

        return out


class lSTM_LAYER(nn.Module):
    def __init__(self, 
                 config: LayerConfig,
                 device: torch.device,
                 dtype: torch.dtype,
                 ):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype

        self.lstm = nn.LSTM(config.embed_dim, config.embed_dim, dropout=config.dropout, batch_first=True, device=device, dtype=dtype)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return out

@component
class HR_AHEAD_BINARY(nn.Module):
    """ Heirarchical recurrent attention with target smoothing"""
    config = ModelConfig
    filepath = __file__
    def __init__(
            self, 
            config: ModelConfig,
            device: torch.device,
            dtype: torch.dtype
    ):
        super().__init__()
        self.config = config 
        self.device = device
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim, device=device, dtype=dtype)

        layers = []
        for i in range(config.n_layers):
            layer_config = LayerConfig(
                process_heads=config.process_heads,
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
            layers.append(HAttnBlock(layer_config, device, dtype))

        self.layers = nn.ModuleList(layers)

        self.head_scorer = MultiHeadSelfAttn(config.embed_dim, config.hidden_dim, 1, config.process_heads, config.dropout, device, dtype)
        self.head_transform = nn.Linear(config.embed_dim, 1, device=device, dtype=dtype)

    def forward(self, batch: Batch) -> Output:
        
        embeddings = self.embed(batch.x)

        B, T, C = embeddings.shape

        out = embeddings
        for layer in self.layers:
            out = layer(out, batch.mask)

        out = out * batch.mask[:, 0, :].unsqueeze(-1)

        scores = self.head_scorer(out, batch.mask)
        dist_logits = self.head_transform(out)
        logits = torch.sum(dist_logits * scores, dim=1)

        probs = torch.nn.functional.sigmoid(logits)

        return Output(logits=logits, probs=probs)

