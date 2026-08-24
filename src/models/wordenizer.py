# legacy: pre-1.4 model, not wired to any experiment. Still uses the
# forward(x, mask) signature; migrate onto models.protocols.Model with a
# canonical batch from data.datasets.types before reuse.

#from ._registry import model_registry
from pydantic import BaseModel, PositiveInt, PositiveFloat
from typing import Literal
import torch
import math
from torch import Tensor
import torch.nn as nn

class ModelConfig(BaseModel):
    model_name: str
    pad_token_id: PositiveInt | Literal[0]
    process_heads: PositiveInt
    word_scope: PositiveInt
    increment: PositiveInt
    n_layers: PositiveInt
    vocab_size: PositiveInt
    embed_dim: PositiveInt
    hidden_dim: PositiveInt
    n_out: PositiveInt
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
        print('Q',  Q.shape) 
        print('K',  K.shape) 
        print('V',  V.shape) 
        print('mask', mask.shape)
        out = nn.functional.scaled_dot_product_attention(Q, K, V, mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1), dropout_p=self.dropout)
        
        out = out.transpose(1, 2).contiguous().flatten(2,-1)
        out = self.head(out)
        return out

class MLP(nn.Module):
    def __init__(
            self,
            config: ModelConfig,
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
                 config: ModelConfig,
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

class Wordenizer(nn.Module):
        def __init__(self, 
                 config: ModelConfig,
                 device: torch.device,
                 dtype: torch.dtype,
                 ):
            super().__init__()
            self.config = config
            self.device = device
            self.dtype = dtype
            self.word_scope = config.word_scope
            self.increment = config.increment
            self.selector_attn = MultiHeadSelfAttn(config.embed_dim, config.hidden_dim, 1, config.process_heads, config.dropout, device, dtype)

        def forward(self, x: Tensor, mask: Tensor) -> Tensor:
            B, T, C = x.shape
            self.n_chunks = 1 + (math.ceil( ( T - self.word_scope ) / self.increment ))

            print('in', x.shape)
            print(mask.shape)
            rng = range(0, T - self.word_scope + self.increment, self.increment) 
            indicies = torch.stack([
                torch.arange( 
                    min(i, T - self.word_scope), 
                    torch.clip(torch.tensor(i + self.word_scope), 0, T),
                    1,) for i in rng 
                ])
            x_index = indicies[None, :, :, None].expand(B, -1, -1, C).flatten(0,1)
            mask_index = indicies[None, :, :, None].expand(B, -1, -1, self.word_scope).flatten(0,1)
            print('x inex', x_index.shape) 
            print('mask inex', mask_index.shape) 

            x_exp = x.unsqueeze(0).expand(indicies.shape[0], -1, -1, -1).flatten(0,1)
            mask_exp = mask.unsqueeze(0).expand(indicies.shape[0], -1, -1, -1).flatten(0,1)

            print('indi', indicies.shape)
            print('mask', mask.shape)
            print('mask exp', mask_exp.shape)
            out = torch.gather(x_exp,dim=1, index=x_index)
            mask = torch.gather(mask_exp, dim=1, index=mask_index)

            print('mask', mask.shape)

            scores = self.selector_attn(out, mask)             
            print('scores', scores.shape)
            print('out', out.shape)
            out = out * scores
            out = out.view(B, self.n_chunks, -1, C)
            print('out view', out.shape)
            out = torch.sum(out, dim=1)
            print(out.shape)
            print('====++')

#@model_registry('Wordenizer')
class Model(nn.Module):
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
        self.config = config 
        self.device = device
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim, device=device, dtype=dtype)
        self.wordenizer = Wordenizer(config, device, dtype)

        layers = []
        for i in range(config.n_layers):
            layers.append(HAttnBlock(config, device, dtype))

        self.layers = nn.ModuleList(layers)

        self.head_scorer = MultiHeadSelfAttn(config.embed_dim, config.hidden_dim, 1, config.process_heads, config.dropout, device, dtype)
        self.head_transform = nn.Linear(config.embed_dim, 1, device=device, dtype=dtype)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        
        embeddings = self.embed(x)
        words = self.wordenizer(embeddings, mask)
        quit()
        B, T, C = words.shape

        out = words 
        mask = mask
        for layer in self.layers:
            out, mask = layer(out, mask)

        out = out * mask[:, 0, :].unsqueeze(-1)

        scores = self.head_scorer(out, mask)
        dist_logits = self.head_transform(out)
        logits = torch.sum(dist_logits * scores, dim=1)

        probs = torch.nn.functional.sigmoid(logits)
        sigma = torch.tensor(float('nan'), dtype=torch.float32)

        return logits, probs, sigma

if __name__ == '__main__':
    embed_dim = 4
    seq_len = 15
    config = ModelConfig(
               model_name='wordenizer',
               pad_token_id=19998,
               word_scope = 5,
               increment = 3,
               process_heads=1,
               n_layers= 4,
               vocab_size= 200_000,
               embed_dim= embed_dim,
               hidden_dim=8,
               n_out= 1,
               dropout= 0.05,
               )
    model = Model(config,
                  torch.device('cpu'),
                  torch.float32,
                  )
    input = torch.randint(1, 20, (2, seq_len))
    mask = torch.ones_like(input)[:, None, :].expand(-1, seq_len, -1).float()
    output = model(input, mask)
    print(*output)

