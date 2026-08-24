# legacy: pre-1.4 model, not wired to any experiment. Still uses the
# forward(x, mask) signature; migrate onto models.protocols.Model with a
# canonical batch from data.datasets.types before reuse.

from pydantic import BaseModel, PositiveInt 
import torch
import torch.nn as nn
from torch import Tensor

def activation_func(x):
    return nn.functional.rms_norm(
        nn.functional.gelu(x),
        (x.size(-1),)
        )

def recursive_mm(x, where_padding):
    n = x.shape[0]
    # case 1: single tensor
    if n < 2:
        if torch.any(where_padding):
            return ((x * 0.0) + torch.eye(x.shape[2], device=x.device)).squeeze(0) # replace with diagonal
        return x.squeeze(0)

    # case 2: > 2 tensors    
    if n > 2:
        return activation_func(
            torch.bmm(
                recursive_mm(x[:n // 2], where_padding[:n // 2]),
                recursive_mm(x[n // 2:], where_padding[n // 2:]),
            )
        )
    # base case: n == 2
    else:
        x0 = x[0] 
        x1 = x[1]
        
        where_both = where_padding[0].unsqueeze(-1).unsqueeze(-1).expand(-1, *x0.shape[1:])
        where_1 = where_padding[1].unsqueeze(-1).unsqueeze(-1).expand(-1, *x1.shape[1:])

        return torch.where(
                    where_both, ((x0 + x1) * 0.0) + torch.eye(x.shape[2], device=x.device), # replace with eye 
                        torch.where(
                            where_1, x0, activation_func(torch.bmm(x0, x1))
                        )
                )

class Projection(nn.Module):
    def __init__(self, in_dim, out_dim, device, dtype):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim, device=device, dtype=dtype),
            nn.GELU(),
        )
    def forward(self, x):
        return self.projection(x) 

class ConfigSchema(BaseModel):
    model_name: str
    vocab_size: PositiveInt 
    eos_token: PositiveInt
    n_layers: PositiveInt
    embed_dim: PositiveInt
    n_out: PositiveInt

class R_RNN_Fast(nn.Module):
    MODEL_NAME = 'mmlp'
    config_schema = ConfigSchema
    def __init__(self, 
                 model_config,
                 device: torch.device,
                 dtype: torch.dtype
                 ):
        super().__init__()
        self.config = model_config
        self.device = device
        self.dtype = dtype 
        
        self.embed = nn.Embedding(model_config.vocab_size, model_config.embed_dim, device=device, dtype=dtype)

        self.embed_projections = nn.ModuleList(
            [Projection(model_config.embed_dim, model_config.embed_dim , device, dtype) for _ in range(model_config.n_layers)]
        ) 

        self.mlps = nn.ModuleList(
            [Projection(model_config.embed_dim, model_config.embed_dim , device, dtype) for _ in range(model_config.n_layers)]
        ) 

        self.normaliser = nn.GELU()

        self.head = nn.Linear(model_config.embed_dim, model_config.n_out, device=device, dtype=dtype)

    def forward(self, x: Tensor):
        indicies = torch.arange(x.shape[1], device=self.device)
        padding_begin = torch.where(x == self.config.eos_token)[1] + 1
        where_padding = torch.zeros_like(x, dtype=torch.bool)
        for i in range(x.shape[0]):
            where_padding[i, :] = indicies > padding_begin[i]

        embeddings = self.embed(x)
        B, T, C = embeddings.shape

        for i in range(self.config.n_layers):
            embed_projections = self.embed_projections[i](embeddings).unsqueeze(-1)
            embed_transforms = embed_projections @ embed_projections.transpose(-1, -2)
            embed_transforms = embed_transforms.permute(1, 0, 2, 3)
                
            embed_mod = recursive_mm(embed_transforms, where_padding.permute(1, 0))
            hidden =  embeddings.unsqueeze(-2) @ embed_mod.unsqueeze(1).expand(-1, T, -1, -1)
            embeddings = embeddings + self.mlps[i](hidden.squeeze(-2))

            embeddings = nn.functional.gelu(embeddings)  
            embeddings = nn.functional.rms_norm(embeddings, (embeddings.size(-1), ))  

        out = torch.mean(self.head(embeddings), dim=1)
        return out

