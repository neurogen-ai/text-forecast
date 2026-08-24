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
    selector_heads: tuple[int, ...]
    process_heads: tuple[int, ...]
    n_layers: PositiveInt
    vocab_size: PositiveInt
    embed_dim: PositiveInt
    hidden_dim: PositiveInt
    n_out: PositiveInt
    dropout: PositiveFloat

class LayerConfig(BaseModel):
    k: PositiveInt
    diff_k: bool
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

        self.process_attn = MultiHeadSelfAttn(config.embed_dim, config.hidden_dim, config.hidden_dim, config.process_heads, config.dropout, device, dtype)

        self.mlp = MLP(config, device, dtype) 
        
    def forward(self, x:Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        B, T, C = x.shape
        x = nn.functional.rms_norm(x, (x.size(-1), ))

        if self.config.diff_k:
            top_k = min(T, self.config.k)
            scores = self.selective_attn(x, mask)  
            scores = scores * mask[:, 0, :].unsqueeze(-1)
            _,  inds = torch.topk(scores.squeeze(-1), top_k, dim=-1)
            inds, _ = torch.sort(inds.long(), dim=1)
            ordered_scores = torch.gather(
                scores, 
                dim=1, 
                index=inds.unsqueeze(-1)
            )
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
                self.process_attn(selected_tokens * ordered_scores, selected_mask)
            )
            return out.contiguous(), selected_mask.contiguous()
        else:
            out = x + self.mlp(
                self.process_attn(x, mask)
            )
            return out.contiguous(), mask

@component
class H_ATTN_SINGLE(nn.Module):
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
                selector_heads=config.selector_heads[i],
                process_heads=config.process_heads[i],
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
            layers.append(HAttnBlock(layer_config, device, dtype))

        self.layers = nn.ModuleList(layers)

        self.head = nn.Linear(config.embed_dim * config.top_k[-1], config.n_out, device=device, dtype=dtype)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        
        embeddings = self.embed(x)

        B, T, C = embeddings.shape

        out = embeddings
        mask = mask
        for layer in self.layers:
            out, mask = layer(out, mask)

        out = out * mask[:, 0, :].unsqueeze(-1)
        out = out.flatten(-2, -1)
        out = self.head(out)
        return out

if __name__ == '__main__':
    test_val = 2 
    
    if test_val == 1:
        torch.manual_seed(2026)
        test_embed = (torch.rand(2,3,2) * 10).long()
        test_scores = (torch.rand(2,3) * 10).long()
        test_mask = (torch.rand(2,3,3) * 10).long()

        _, test_inds = torch.topk(test_scores, dim=1, k=2)
        print(test_mask)
        print(test_mask.shape)
        print(test_inds)
        print(test_inds.shape)
        test_gather = torch.gather(test_mask, dim=1, index=test_inds.unsqueeze(-1).expand(-1, -1, 3))
        print(test_gather)
    
    if test_val == 3:
        config = ModelConfig(
                model_name = 'test',
                pad_token = 0,
                outer_heads = 2,
                top_k = [4,1],
                selector_heads = [3,3],
                process_heads = [3,3],
                n_layers = 2,
                vocab_size = 4 ,
                embed_dim = 3,
                hidden_dim = 10,
                n_out = 5,
            )
        model = H_ATTN(config, 'cpu', torch.float32)
        input = torch.ones(7, 5)
        mask = torch.ones(7, 5, 5)
        out = model(input.long(), mask)
        print(out.shape)
