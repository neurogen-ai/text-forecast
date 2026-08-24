# legacy: pre-1.4 model, not wired to any experiment. Still uses the
# forward(x, mask) signature; migrate onto models.protocols.Model with a
# canonical batch from data.datasets.types before reuse.

#from ._registry import model_registry
from pydantic import BaseModel, PositiveInt, PositiveFloat
from typing import Literal
import torch
from torch import Tensor
import torch.nn as nn

class ModelConfig(BaseModel):
    model_name: str
    pad_token_id: PositiveInt | Literal[0] 
    n_heads: PositiveInt
    n_layers: PositiveInt
    vocab_size: PositiveInt
    embed_dim: PositiveInt
    hidden_dim: PositiveInt
    n_abstractions: PositiveInt 
    abstraction_depth: PositiveInt
    n_out: PositiveInt
    n_params_out: PositiveInt | Literal[0]
    out_seq_len: PositiveInt
    dropout: PositiveFloat | Literal[0]

class MultiHeadSelfAttn(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, 'Embed dim must split evenly across n heads'
        self.embed_dim: int = embed_dim
        self.n_heads: int = n_heads
        self.dropout: float = dropout
        self.head_dim: int = embed_dim // n_heads
        self.QKV_proj: nn.Linear = nn.Linear(embed_dim, embed_dim * 3, device=device, dtype=dtype)
        self.out_proj: nn.Linear = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype) 

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv: Tensor = self.QKV_proj.forward(x)
        q, k, v = torch.split(qkv, self.embed_dim, dim=2)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1,2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1,2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1,2)
        
        if self.training:
            out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout)
        else:
            out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj.forward(out)
        return out

class SelectiveAttn(nn.Module):
    def __init__(
        self,
        embed_dim: PositiveInt,
        n_heads: PositiveInt,
        n_out: PositiveInt,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, 'Embed dim must split evenly across n heads'
        self.embed_dim: int = embed_dim
        self.n_heads: int = n_heads
        self.dropout: float = dropout
        self.head_dim: int = embed_dim // n_heads
        self.QKV_proj: nn.Linear = nn.Linear(embed_dim, embed_dim * 3, device=device, dtype=dtype)
        self.out_proj: nn.Linear = nn.Linear(embed_dim, n_out, device=device, dtype=dtype) 

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv: Tensor = self.QKV_proj.forward(x)
        q, k, v = torch.split(qkv, self.embed_dim, dim=2)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1,2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1,2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1,2)
        
        if self.training:
            out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout)
        else:
            out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj.forward(out)
        return out

class MLP(nn.Module):
    def __init__(
            self,
            in_dim: PositiveInt,
            embed_dim: PositiveInt,
            hidden_dim: PositiveInt,
            dropout: float,
            device: torch.device,
            dtype: torch.dtype,
    ):
        super().__init__() 
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

class TransformerBlock(nn.Module):
    def __init__(self, 
                 embed_dim: PositiveInt,
                 hidden_dim: PositiveInt,
                 n_heads: PositiveInt,
                 dropout: PositiveFloat | Literal[0],
                 device: torch.device,
                 dtype: torch.dtype,
                 ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.attn = MultiHeadSelfAttn(embed_dim, n_heads, dropout, device, dtype) 
        self.mlp = MLP(embed_dim, embed_dim, hidden_dim, dropout, device, dtype) 

    def forward(self, x:Tensor, mask: Tensor) -> Tensor:
        B, T, C = x.shape
        x = x + self.attn(
                    nn.functional.rms_norm(x, (x.size(-1), )),
                    mask
                )

        x = x + self.mlp(
                    nn.functional.rms_norm(x, (x.size(-1), )),
                )

        return x

class AbstractorBlock(nn.Module):
    def __init__(self, 
                 embed_dim: PositiveInt,
                 hidden_dim: PositiveInt,
                 n_heads: PositiveInt,
                 n_abstractions: PositiveInt,
                 dropout: PositiveFloat | Literal[0],
                 device: torch.device,
                 dtype: torch.dtype,
                 ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.selective_attn = SelectiveAttn(embed_dim, n_heads, n_abstractions, dropout, device, dtype) 
        self.mlp = MLP(embed_dim, embed_dim, hidden_dim, dropout, device, dtype) 
        
    def batch_mult(self, x: Tensor, scores: Tensor) -> Tensor:
        return torch.bmm(scores.transpose(1,2), x) # [B, n_abstractions, T] @ [B, T, C] -> B, n_abstractions, C

    def residual_batch_mult(self, x: Tensor, scores: Tensor) -> Tensor:
        return x + self.batch_mult(x, scores)
    
    def forward(self, x:Tensor, mask: Tensor) -> Tensor:
        B, T, C = x.shape
        x_norm = nn.functional.rms_norm(x, (x.size(-1), ))
        scores = self.selective_attn(x_norm, mask) # B, T, n_abstractions

        if T == scores.size(-1):
            x = x + self.residual_batch_mult(x_norm, scores)
        else:
            x = self.batch_mult(x_norm, scores)

        x = x + self.mlp(nn.functional.rms_norm(x, (x.size(-1), )))
        return x

#@model_registry
class AbstractorDLM(nn.Module):
    filepath = __file__
    config_schema = ModelConfig
    def __init__(
            self, 
            config: ModelConfig,
            device: torch.device,
            dtype: torch.dtype
    ):
        super().__init__()
        self.config = config 
        self.device = device
        self.dtype = dtype
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim, device=device, dtype=dtype)
        self.t_proj = MLP(1, config.embed_dim, config.embed_dim, config.dropout, device, dtype) 

        self.transformer_blocks = nn.ModuleList(
                [TransformerBlock(
                    embed_dim=config.embed_dim,
                    hidden_dim=config.hidden_dim,
                    n_heads=config.n_heads,
                    dropout=config.dropout,
                    device=device,
                    dtype=dtype,
                    ) for _ in range(config.n_layers)]
                )

        self.abstractor_blocks = nn.ModuleList(
                [AbstractorBlock(
                    embed_dim=config.embed_dim,
                    hidden_dim=config.hidden_dim,
                    n_heads=config.n_heads,
                    n_abstractions=config.n_abstractions,
                    dropout=config.dropout,
                    device=device,
                    dtype=dtype,
                    ) for _ in range(config.abstraction_depth)]
                )

        self.concretion_head = AbstractorBlock(
                    embed_dim=config.embed_dim,
                    hidden_dim=config.hidden_dim,
                    n_heads=config.n_heads,
                    n_abstractions=config.out_seq_len,
                    dropout=config.dropout,
                    device=device,
                    dtype=dtype,
                    )
        self.head = nn.Linear(config.embed_dim, config.vocab_size, device=device, dtype=dtype)
        
    def forward(self, x: Tensor, mask: Tensor, t: Tensor) -> dict[str, Tensor]:
        
        embeddings = self.embed(x)
        B, T, C = embeddings.shape
        sequence_mask = mask.clone()

        out = embeddings
        for abstractor in self.abstractor_blocks:
            for layer in self.transformer_blocks:
                out = layer(out, mask)
            out = abstractor(out, mask) 
            mask = torch.ones(B, self.config.n_abstractions, self.config.n_abstractions)
         
        out = self.concretion_head(out, mask)

        out = out * sequence_mask[:, 1, :].unsqueeze(-1)

        logits = self.head(out)
        probs = torch.nn.functional.softmax(logits, dim=2)

        return {
                'logits': logits,
                'probs': probs,
                'embeddings': out
                }

    def generate(self, prompt: Tensor, max_len: int):
        out = []
        probs = []
        tokens = []
        for i in range(max_len):
            B, T = prompt.shape
            mask = torch.ones_like(prompt).unsqueeze(-1).expand(B, T, T).float()
            logits, probs = self.forward(prompt, mask)
            tokens = torch.argmax(probs, dim=2)
            next = tokens[:, -1]
            out += next.tolist()
            prompt = torch.cat([prompt, next.unsqueeze(0)], dim=1)

        return out

if __name__ == '__main__':
    test_val = 3 
    
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
        seq_len = 5
        batch_size = 7
        input = torch.ones(batch_size, seq_len)
        mask = torch.ones(batch_size, seq_len, seq_len)

        config = ModelConfig(
                model_name = 'test',
                pad_token_id = 0,
                n_heads = 2,
                n_layers = 2,
                n_abstractions=2,
                abstraction_depth=1,
                vocab_size = 4 ,
                embed_dim = 4,
                hidden_dim = 10,
                n_out = 4,
                n_params_out = 1,
                out_seq_len = seq_len,
                dropout=0.01,
            )
        model = AbstractorDLM(config, torch.device('cpu'), torch.float32)
        out = model(input.long(), mask, 1)
        for k, v in out.items():
            print(k, v.shape)

        model = model.compile(fullgraph=True)
        print('model compiles')
