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
    top_k: tuple[int, ...]
    r_layers: tuple[int, ...]
    selector_heads: tuple[int, ...]
    process_heads: tuple[int, ...]
    n_layers: PositiveInt
    vocab_size: PositiveInt
    embed_dim: PositiveInt
    hidden_dim: PositiveInt
    n_out: PositiveInt
    n_params_out: int
    dropout: PositiveFloat

class LayerConfig(BaseModel):
    k: PositiveInt
    diff_k: bool
    r_layer: bool
    selector_heads: PositiveInt
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

class SelectiveAttn(nn.Module):
    def __init__(self,
                 config: LayerConfig,
                 device: torch.device,
                 dtype: torch.dtype,
                 ):
        super().__init__()
        self.relevance_selector = MultiHeadSelfAttn(config.embed_dim, config.hidden_dim, config.hidden_dim, config.selector_heads, config.dropout, device=device, dtype=dtype)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, 1, device=device, dtype=dtype),
            nn.GELU(),
        )

    def forward(self, x:Tensor, mask: Tensor) -> Tensor:
        out = self.relevance_selector(x, mask)
        out = nn.functional.softmax(out, dim=-1) 
        out = self.head(out)
        return out

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

        if config.diff_k:
            self.selective_attn = SelectiveAttn(config, device, dtype)

        if self.config.r_layer:
            self.r_net = nn.GRU(config.embed_dim, config.embed_dim, dropout=config.dropout, batch_first=True, device=device, dtype=dtype)

        self.process_attn = MultiHeadSelfAttn(config.embed_dim, config.hidden_dim, config.hidden_dim, config.process_heads, config.dropout, device, dtype)

        self.mlp = MLP(config, device, dtype) 
        
    def forward(self, x:Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        B, T, C = x.shape
        x = nn.functional.rms_norm(x, (x.size(-1), ))

        if self.config.diff_k:
            top_k = min(T, self.config.k)
            scores = self.selective_attn(x, mask)  
            scores = scores * mask[:, 0, :].unsqueeze(-1)
            x = x * scores
            _,  inds = torch.topk(scores.squeeze(-1), top_k, dim=-1)
            inds, _ = torch.sort(inds.long(), dim=1)
#            ordered_scores = torch.gather(
#                scores, 
#                dim=1, 
#                index=inds.unsqueeze(-1)
#            )

            selected_tokens = torch.gather(
                x, 
                dim=1, 
                index=inds.unsqueeze(-1).expand(-1, -1, C)
            )
             
            selected_mask = torch.gather(
                mask,
                dim=1,
                index=inds.unsqueeze(-1).expand(-1, -1, top_k) 
            )
            out = selected_tokens + self.mlp(
                self.process_attn(selected_tokens, selected_mask)
            )
            out = out.contiguous()
            mask = selected_mask.contiguous()
        else:
            out = x + self.mlp(
                self.process_attn(x, mask)
            )

            out = out.contiguous()
        
        if self.config.r_layer:
            r_out, _ = self.r_net(out)
            out = out + r_out
            #out, _ = self.r_net(out)

        return out, mask

@component
class H_R_Smooth(nn.Module):
    """ Heirarchical recurrent attention with target smoothing"""
    config_schema = ModelConfig
    def __init__(
            self, 
            config: ModelConfig,
            device: torch.device,
            dtype: torch.dtype
    ):
        super().__init__()
        assert len(config.process_heads) == config.n_layers
        assert len(config.selector_heads) == config.n_layers
        assert len(config.top_k) == config.n_layers
        self.config = config 

        self.embed = nn.Embedding(config.vocab_size, config.embed_dim, device=device, dtype=dtype)
        top_k_diffs: list[bool] = [False for _ in range(config.n_layers)] 
        for i in range(config.n_layers - 1):
            if  config.top_k[i] > config.top_k[i + 1]:
                top_k_diffs[i + 1] = True
            assert config.top_k[i] >= config.top_k[i + 1], 'top_k cannot increase in subsequent layers'

        layers = []
        for i in range(config.n_layers):
            layer_config = LayerConfig(
                k=config.top_k[i],
                diff_k = top_k_diffs[i],
                r_layer = True if self.config.r_layers[i] else False,
                selector_heads=config.selector_heads[i],
                process_heads=config.process_heads[i],
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
            layers.append(HAttnBlock(layer_config, device, dtype))

        self.layers = nn.ModuleList(layers)

        # head projects to n_out + n_params_out for confidence (target smoothing values)
        self.head = nn.Linear(config.embed_dim * config.top_k[-1], config.n_out + config.n_params_out, device=device, dtype=dtype)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        
        embeddings = self.embed(x)

        B, T, C = embeddings.shape

        out = embeddings
        mask = mask
        for layer in self.layers:
            out, mask = layer(out, mask)

        out = out * mask[:, 0, :].unsqueeze(-1)
        out = out.flatten(-2, -1)
        logits = self.head(out)
        if self.config.n_params_out > 0:
            out, sigma = logits[:, :-self.config.n_params_out], logits[:, -self.config.n_params_out:]
            probs = torch.softmax(out, dim=-1)
            sigma = nn.functional.sigmoid(sigma)
        else:
            out = logits
            probs = torch.nn.functional.sigmoid(out)
            sigma = torch.tensor(float('nan'), dtype=torch.float32)

        return logits, probs, sigma

