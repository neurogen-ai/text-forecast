# legacy: pre-1.4 model, not wired to any experiment. Still uses the
# forward(x, mask) signature; migrate onto models.protocols.Model with a
# canonical batch from data.datasets.types before reuse.

from pydantic import BaseModel, PositiveInt 
import torch
import torch.nn as nn
from torch import Tensor

class ConfigSchema(BaseModel):
    model_name: str
    vocab_size: PositiveInt 
    eos_token: PositiveInt
    n_layers: PositiveInt
    embed_dim: PositiveInt
    attention_dim: PositiveInt
    n_out: PositiveInt

class R_RNN(nn.Module):
    MODEL_NAME = 'r_rnn'
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

        self.attention_proj = nn.Sequential(
            nn.Linear(model_config.attention_dim, model_config.embed_dim, device=device, dtype=dtype),
            nn.GELU(),
        )
        
        self.embed_proj = nn.Sequential(
            nn.Linear(model_config.embed_dim, model_config.attention_dim, device=device, dtype=dtype),
            nn.GELU(),
        )

        self.normaliser = nn.GELU()

        self.head = nn.Linear(model_config.attention_dim, model_config.n_out, device=device, dtype=dtype)

    def eos_attention_escape(self, attention: Tensor, next_attention: Tensor, eos_one_hot_t: Tensor):
        """ Attention addition for torch.cond (true_fn) call """
        next_attention += attention.squeeze(-1) * eos_one_hot_t.unsqueeze(-1)
    
    def no_eos_func(self, attention: Tensor, next_attention: Tensor, eos_one_hot_t: Tensor):
        """ Attention addition for torch.cond (false_fn) call """
        pass

    def forward(self, x: Tensor):
        eos_one_hot = torch.zeros_like(x)
        max_eos_idx = torch.max(torch.where(x == self.config.eos_token)[1])
        eos_one_hot[x == self.config.eos_token] = 1
        embeddings = self.embed(x)
        B, T, C = embeddings.shape
        attention = torch.ones((B, self.config.attention_dim, 1), device=self.device, dtype=self.dtype)
        for layer in range(self.config.n_layers):
            embed_projections = self.embed_proj(embeddings).unsqueeze(-1)
            embed_transforms = embed_projections @ embed_projections.transpose(-1, -2)
            next_attention = torch.zeros_like(attention.squeeze(-1))
            for t in range(max_eos_idx):
                attention = embed_transforms[:, t, :, :] @ attention
                attention = nn.functional.rms_norm(attention, (attention.shape[-1], ))
                next_attention += attention.squeeze(-1) * eos_one_hot[:, t].unsqueeze(-1)

            attention = next_attention.unsqueeze(-1).clone()
            attention_projection = self.attention_proj(attention.squeeze(-1)).unsqueeze(-1)
            attention_transformation = attention_projection @ attention_projection.transpose(-1, -2)
            embeddings = attention_transformation.unsqueeze(1).expand(-1, T, -1, -1) @ embeddings.unsqueeze(-1)
            embeddings = embeddings.squeeze(-1)

        out = self.head(attention.squeeze(-1)) 
        return out

