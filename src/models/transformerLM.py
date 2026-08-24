from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, PositiveFloat, PositiveInt
from torch import Tensor

from utils import component


class ModelConfig(BaseModel):
    model_name: str
    pad_token_id: int
    n_heads: int
    n_layers: PositiveInt
    vocab_size: PositiveInt
    embed_dim: PositiveInt
    hidden_dim: PositiveInt
    n_out: PositiveInt
    dropout: PositiveFloat


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
        assert embed_dim % n_heads == 0, "Embed dim must split evenly across n heads"
        self.embed_dim: int = embed_dim
        self.n_heads: int = n_heads
        self.dropout: float = dropout
        self.head_dim: int = embed_dim // n_heads
        self.QKV_proj: nn.Linear = nn.Linear(
            embed_dim, embed_dim * 3, device=device, dtype=dtype
        )
        self.out_proj: nn.Linear = nn.Linear(
            embed_dim, embed_dim, device=device, dtype=dtype
        )

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        B, T, C = x.shape
        qkv: Tensor = self.QKV_proj.forward(x)
        q, k, v = torch.split(qkv, self.embed_dim, dim=2)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=attn_mask is None and self.training,
            dropout_p=self.dropout,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj.forward(out)
        return out


class MLP(nn.Module):
    def __init__(
        self,
        embed_dim: PositiveInt,
        hidden_dim: PositiveInt,
        dropout: PositiveFloat | Literal[0],
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
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
        self.mlp = MLP(embed_dim, hidden_dim, dropout, device, dtype)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        B, T, C = x.shape

        x = x + self.attn(nn.functional.rms_norm(x, (x.size(-1),)), attn_mask)

        x = x + self.mlp(
            nn.functional.rms_norm(x, (x.size(-1),)),
        )

        return x


@component
class TransformerLM(nn.Module):
    filepath = __file__
    config_schema = ModelConfig

    def __init__(self, config: ModelConfig, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype
        self.embed = nn.Embedding(
            config.vocab_size, config.embed_dim, device=device, dtype=dtype
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    config.embed_dim,
                    config.hidden_dim,
                    config.n_heads,
                    config.dropout,
                    device,
                    dtype,
                )
                for _ in range(config.n_layers)
            ]
        )

        self.head = nn.Linear(
            config.embed_dim, config.n_out, device=device, dtype=dtype
        )

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        # mask is (B, T) bool key-padding.
        embeddings = self.embed(x)
        B, T, C = embeddings.shape
        if self.training:
            causal_mask = ~torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=self.device), diagonal=1
            )
            # (T, T) causal AND (B, 1, T) padding -> (B, 1, T, T) for SDPA.
            attn_mask = causal_mask.unsqueeze(0) & mask[:, None, :]
        else:
            attn_mask = mask[:, None, None, :]

        out = embeddings
        for layer in self.layers:
            out = layer(out, attn_mask)

        out = out * mask.unsqueeze(-1)
        out = nn.functional.rms_norm(out, (out.size(-1),))

        logits = self.head(out)
        probs = torch.nn.functional.softmax(logits, dim=2)

        return logits, probs

    def generate(self, prompt: Tensor, max_len: int):
        out = []
        probs = []
        tokens = []
        for i in range(max_len):
            B, T = prompt.shape
            mask = torch.ones_like(prompt).bool()
            logits, probs = self.forward(prompt, mask)
            tokens = torch.argmax(probs, dim=2)
            next = tokens[:, -1]
            out += next.tolist()
            prompt = torch.cat([prompt, next.unsqueeze(0)], dim=1)

        return out


if __name__ == "__main__":
    test_val = 2

    if test_val == 1:
        torch.manual_seed(2026)
        test_embed = (torch.rand(2, 3, 2) * 10).long()
        test_scores = (torch.rand(2, 3) * 10).long()
        test_mask = (torch.rand(2, 3, 3) * 10).long()

        _, test_inds = torch.topk(test_scores, dim=1, k=2)
        print(test_mask)
        print(test_mask.shape)
        print(test_inds)
        print(test_inds.shape)
        test_gather = torch.gather(
            test_mask, dim=1, index=test_inds.unsqueeze(-1).expand(-1, -1, 3)
        )
        print(test_gather)

    if test_val == 3:
        config = ModelConfig(
            model_name="test",
            pad_token=0,
            outer_heads=2,
            top_k=[4, 1],
            selector_heads=[3, 3],
            process_heads=[3, 3],
            n_layers=2,
            vocab_size=4,
            embed_dim=3,
            hidden_dim=10,
            n_out=5,
        )
        model = H_ATTN(config, "cpu", torch.float32)
        input = torch.ones(7, 5)
        mask = torch.ones(7, 5, 5)
        out = model(input.long(), mask)
        print(out.shape)
