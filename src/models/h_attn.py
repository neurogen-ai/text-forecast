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
    outer_heads: PositiveInt
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
    outer_heads: PositiveInt
    k: PositiveInt
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
        
        out = out.transpose(1, 2).contiguous().flatten(-2)
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
        self.selective_attn = nn.ModuleList(
                [ SelectiveAttn(config, device, dtype) for _ in range(config.outer_heads) ]
        )
        self.process_attn = nn.ModuleList(
                [ MultiHeadSelfAttn(config.embed_dim, config.hidden_dim, config.hidden_dim, config.process_heads, config.dropout, device, dtype) for _ in range(config.outer_heads) ]
        )
        self.mlp = nn.ModuleList(
                [ MLP(config, device, dtype) for _ in range(config.outer_heads) ]
        ) 
    def forward(self, x:Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        OB, B, T, C = x.shape
        x = nn.functional.rms_norm(x, (x.size(-1), ))
        out = torch.zeros(OB, B, self.config.k, C, device=self.device, dtype=self.dtype)
        mask_out = torch.zeros(OB, B, self.config.k, self.config.k, device=self.device, dtype=torch.bool)
        for i in range(self.config.outer_heads):
            scores = self.selective_attn[i](x[i], mask[i])  
            scores = scores * mask[i, :, 0, :].unsqueeze(-1)
            _,  inds = torch.topk(scores.squeeze(-1), self.config.k, dim=-1)
            inds, _ = torch.sort(inds.long(), dim=1)
            ordered_scores = torch.gather(
                scores, 
                dim=1, 
                index=inds.unsqueeze(-1)
            )
            selected_tokens = torch.gather(
                x[i], 
                dim=1, 
                index=inds.unsqueeze(-1).expand(-1, -1, C)
            )
             
            selected_mask = torch.gather(
                mask[i],
                dim=1,
                index=inds.unsqueeze(-1).expand(-1, -1, self.config.k)
            )

            mask_out[i] = mask_out[i] + selected_mask.clone()
            
            out[i] = out[i] + selected_tokens + self.mlp[i](
                self.process_attn[i](selected_tokens * ordered_scores, selected_mask)
            )
        return out.contiguous(), mask_out.contiguous()

@component
class H_ATTN(nn.Module):
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
        
        layers = []
        for i in range(config.n_layers):
            layer_config = LayerConfig(
                outer_heads=config.outer_heads,
                k=config.top_k[i],
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

        out = embeddings.unsqueeze(0).expand(self.config.outer_heads, -1, -1, -1)
        mask = mask.unsqueeze(0).expand(self.config.outer_heads, -1, -1, -1)
        for layer in self.layers:
            out, mask = layer(out, mask)
        
        out = out * mask[:, :, 0, :].unsqueeze(-1)
        out = out.flatten(-2, -1)
        out = self.head(out)
        return torch.mean(out, dim=0)    

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
