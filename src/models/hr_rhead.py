# legacy: pre-1.4 model, not wired to any experiment. Still uses the
# forward(x, mask) signature; migrate onto models.protocols.Model with a
# canonical batch from data.datasets.types before reuse.

from pydantic import BaseModel, PositiveInt, PositiveFloat
from utils import component
import torch
from torch import Tensor
import torch.nn as nn

class ModelConfig(BaseModel):
    model_name: str
    pad_token_id: int
    r_layers: tuple[int, ...]
    process_heads: tuple[int, ...]
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

        return out, mask


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
class HR_RHEAD_BINARY(nn.Module):
    """ Heirarchical recurrent attention with target smoothing"""
    config_schema = ModelConfig
    filepath = __file__
    def __init__(
            self, 
            config: ModelConfig,
            device: torch.device,
            dtype: torch.dtype
    ):
        super().__init__()
        assert len(config.process_heads) == config.n_layers
        self.config = config 
        self.device = device
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim, device=device, dtype=dtype)

        layers = []
        for i in range(config.n_layers):
            layer_config = LayerConfig(
                process_heads=config.process_heads[i],
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
            layers.append(HAttnBlock(layer_config, device, dtype))

            if self.config.r_layers[i]:
                r_layer = lSTM_LAYER(config, device, dtype)
                layers.append(r_layer)    
                
        self.layers = nn.ModuleList(layers)

        self.head = nn.LSTM(config.embed_dim, config.n_out, batch_first=True, device=device, dtype=dtype)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        
        embeddings = self.embed(x)

        B, T, C = embeddings.shape

        out = embeddings
        mask = mask
        for layer in self.layers:
            out, mask = layer(out, mask)

        out = out * mask[:, 0, :].unsqueeze(-1)
        index = torch.arange(0, T, device=self.device).unsqueeze(0).expand(B, -1,) 
        index = index * mask[:, 0, :]
        t_end = torch.argmax(index, dim=1)
        logits, _ = self.head(out)
        logits = logits.squeeze(-1)
        logits = torch.gather(logits, dim=1, index=t_end[None, :]).squeeze(0)

        out = logits
        probs = torch.nn.functional.sigmoid(out)
        sigma = torch.tensor(float('nan'), dtype=torch.float32)

        return logits, probs, sigma

