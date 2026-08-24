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
    process_heads: int
    n_layers: PositiveInt
    abstracted_seq_len: PositiveInt
    abstraction_heads: PositiveInt
    n_abstractions: PositiveInt
    n_forward: PositiveInt
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
class AbstractorLM(nn.Module):
    """ Multi level sequence abstraction """
    config_schema = ModelConfig
    filepath = __file__
    def __init__(
            self, 
            config: ModelConfig,
            device: torch.device,
            dtype: torch.dtype
    ):
        super().__init__()
        self.abstracted_seq_len = config.abstracted_seq_len
        self.config = config 
        self.device = device
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim, device=device, dtype=dtype)
        
        abstraction_scorers = []
        process_layers = []

        for i in range(config.n_abstractions):
            layer = MultiHeadSelfAttn(
                    config.embed_dim, 
                    config.hidden_dim, 
                    config.abstraction_heads,
                    config.abstraction_heads, 
                    config.dropout,
                    device=self.device, 
                    dtype=dtype
                    )
            abstraction_scorers.append(layer)

            for i in range(config.n_layers):
                layer_config = LayerConfig(
                    process_heads=config.process_heads,
                    embed_dim=config.embed_dim,
                    hidden_dim=config.hidden_dim,
                    dropout=config.dropout,
                )
                process_layers.append(HAttnBlock(layer_config, device=device, dtype=dtype))
                
        self.process_layers = nn.ModuleList(process_layers)
        self.abstraction_scorers = nn.ModuleList(abstraction_scorers)

        self.head_transform = nn.Linear(config.embed_dim, config.n_out, device=device, dtype=dtype)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        embeddings = self.embed(x)

        B, T, C = embeddings.shape
        if self.train:
            causal_mask = ~torch.triu(torch.ones(T, T, dtype=torch.bool, device=self.device), diagonal=1)
            mask = mask * causal_mask

        out = embeddings
        for i in range(self.config.n_abstractions):
            for j in range(self.config.n_layers):
                out = self.process_layers[i + j](out, mask)
            scores = self.abstraction_scorers[i](out, mask)

            _, inds = torch.topk(scores, k=min(self.abstracted_seq_len, T), dim=1)
            out = out.unsqueeze(-1).expand(B, -1, -1, self.config.abstraction_heads) 
            out = out * scores.unsqueeze(-2)
            inds = inds.unsqueeze(-2).expand(B, -1, self.config.embed_dim, -1)
            out = torch.gather(out, dim=1, index=inds).squeeze(-1)
            out = torch.sum(out, dim=-1)
            mask = torch.ones((B, self.abstracted_seq_len, self.abstracted_seq_len), device=self.device)
            if self.train:
                causal_mask = ~torch.triu(torch.ones(self.abstracted_seq_len, self.abstracted_seq_len, dtype=torch.bool, device=self.device), diagonal=1)
                mask = mask * causal_mask


        logits = self.head_transform(out)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        return logits, probs

    def generate(self, prompt: Tensor, max_len: int):
        out = []
        for i in range(max_len):
            B, T = prompt.shape
            mask = torch.ones_like(prompt).unsqueeze(-1).expand(B, T, T).float()
            logits, probs = self.forward(prompt, mask)
            tokens = torch.argmax(probs, dim=2)
            next = tokens[:, :self.config.n_forward]
            out += next.tolist()
            prompt = torch.cat([prompt, next], dim=1)

        return out

if __name__ == '__main__':
    embed_dim = 4
    seq_len = 110
    config = ModelConfig(
               model_name='wordenizer',
               pad_token_id=19998,
               process_heads=1,
               n_layers= 4,
               n_abstractions=2,
               abstraction_heads=4,
               abstracted_seq_len=100,
               vocab_size= 200_000,
               embed_dim= embed_dim,
               hidden_dim=8,
               n_out= 1,
               dropout= 0.05,
               )
    model = Abstractor(config,
                  torch.device('cpu'),
                  torch.float32,
                  )
    input = torch.randint(1, 20, (2, seq_len))
    mask = torch.ones_like(input)[:, None, :].expand(-1, seq_len, -1).float()
    output = model(input, mask)
    print(*output)

