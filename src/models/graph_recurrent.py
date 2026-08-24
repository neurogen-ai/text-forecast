# legacy: pre-1.4 model, not wired to any experiment. Batch-style but still
# assumes the old square (B,T,T) mask; migrate onto models.protocols.Model
# with the plain (B,T) key-padding mask from data.datasets.types before reuse.

from typing import Literal, NamedTuple, Protocol

import torch
import torch.nn as nn
from .protocols import Model
from pydantic import BaseModel, PositiveFloat, PositiveInt
from torch import Tensor


class ModelConfig(BaseModel):
    n_heads: int
    n_layers: PositiveInt
    vocab_size: PositiveInt
    embed_dim: PositiveInt
    hidden_dim: PositiveInt
    n_out: PositiveInt
    dropout: float


class BatchInput(Protocol):
    x: Tensor
    mask: Tensor


class Output(NamedTuple):
    logits: Tensor
    probs: Tensor


class MultiHeadSelfAttn(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_out: int,
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
            embed_dim, n_out, device=device, dtype=dtype
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv: Tensor = self.QKV_proj.forward(x)
        q, k, v = torch.split(qkv, self.embed_dim, dim=2)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.unsqueeze(1), dropout_p=self.dropout
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

        self.attn = MultiHeadSelfAttn(
            embed_dim, n_heads, embed_dim, dropout, device, dtype
        )
        self.mlp = MLP(embed_dim, hidden_dim, dropout, device, dtype)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B, T, C = x.shape
        x = x + self.attn(nn.functional.rms_norm(x, (x.size(-1),)), mask)

        x = x + self.mlp(
            nn.functional.rms_norm(x, (x.size(-1),)),
        )

        return x


class GraphRecurrent(nn.Module, Model[ModelConfig, BatchInput, Output]):
    config = ModelConfig

    def __init__(self, config: ModelConfig, device: torch.device, dtype: torch.dtype):
        super().__init__()
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

        self.head_scorer = MultiHeadSelfAttn(
            config.embed_dim,
            config.n_heads,
            1,
            dropout=config.dropout,
            device=device,
            dtype=dtype,
        )
        self.head = nn.Linear(
            config.embed_dim, config.n_out, device=device, dtype=dtype
        )

    def forward(self, batch: BatchInput) -> Output:

        embeddings = self.embed(batch.x)
        B, T, C = embeddings.shape

        out = embeddings
        for layer in self.layers:
            out = layer(out, batch.mask)

        mask = batch.mask[:, 0, :].unsqueeze(-1)
        # out = out * mask
        out = nn.functional.rms_norm(out, (out.size(-1),))
        # print(torch.sum(mask[1],dim=0))
        # print(torch.sum(batch.x[1] != 1999, dim=0))
        # print()
        raw_scores = self.head_scorer(out, batch.mask)
        masked_scores = raw_scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.nn.functional.softmax(masked_scores, dim=1)

        logits = torch.sum(self.head(out) * scores, dim=1)
        probs = torch.nn.functional.sigmoid(logits)
        return Output(logits=logits, probs=probs)


if __name__ == "__main__":
    from typing import NamedTuple

    test_val = 3
    if test_val == 0:
        config = ModelConfig(
            n_heads=1,
            n_layers=8,
            vocab_size=201_088,
            embed_dim=32,
            hidden_dim=32 * 4,
            n_out=1,
            dropout=0.0,
        )
        model = TransformerClass(config, "cpu", torch.float32)

        count = 0
        embedding_params = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                for param in module.parameters(recurse=False):
                    embedding_params += param.numel()

                continue  # Skip embedding layers

            # Count parameters in non-embedding modules
            for param in module.parameters(recurse=False):
                count += param.numel()

        print(f"N Parameters: {count:,}")
        print(f"N embed params: {embedding_params:,}")
        print(f"total: {count + embedding_params:,}")
        print(sum(p.numel() for p in model.parameters()))

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
            n_heads=1,
            n_layers=2,
            vocab_size=4,
            embed_dim=3,
            hidden_dim=10,
            dropout=0.05,
            n_out=5,
        )
        model = GraphRecurrent(config, "cpu", torch.float32)
        input = torch.ones(7, 5)
        mask = torch.ones(7, 5, 5)

        class Batch(NamedTuple):
            x: Tensor
            mask: Tensor

        out = model(Batch(x=input.long(), mask=mask))
        print(out.probs.shape)
