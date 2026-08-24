# legacy: pre-1.4 model, not wired to any experiment. Batch-style but still
# assumes the old square (B,T,T) mask; migrate onto models.protocols.Model
# with the plain (B,T) key-padding mask from data.datasets.types before reuse.

from typing import NamedTuple, Protocol

import torch
import torch.nn as nn
from pydantic import BaseModel, PositiveFloat, PositiveInt
from torch import Tensor

from utils import component

from .protocols import Model


class ModelConfig(BaseModel):
    process_heads: int
    n_layers: PositiveInt
    abstracted_seq_len: PositiveInt
    abstraction_heads: PositiveInt
    n_abstractions: PositiveInt
    vocab_size: PositiveInt
    embed_dim: PositiveInt
    hidden_dim: PositiveInt
    n_out: PositiveInt
    dropout: PositiveFloat


class BatchInput(Protocol):
    x: Tensor
    mask: Tensor


class Output(NamedTuple):
    logits: Tensor
    probs: Tensor


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
        self.embed_dim = dim_in
        self.head_dim: int = self.embed_dim // n_heads
        self.n_heads = n_heads
        self.dropout = dropout
        self.QKV_proj = nn.Linear(dim_in, dim_in * 3, device=device, dtype=dtype)
        self.head = nn.Linear(dim_in, dim_out, device=device, dtype=dtype)

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

        out = self.head.forward(out)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
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
            nn.Linear(config.embed_dim, config.hidden_dim, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.embed_dim, device=device, dtype=dtype),
            nn.Dropout(p=config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class HAttnBlock(nn.Module):
    def __init__(
        self,
        config: LayerConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype

        self.process_attn = MultiHeadSelfAttn(
            config.embed_dim,
            config.hidden_dim,
            config.embed_dim,
            config.process_heads,
            config.dropout,
            device,
            dtype,
        )

        self.mlp = MLP(config, device, dtype)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        B, T, C = x.shape
        x = nn.functional.rms_norm(x, (x.size(-1),))
        out = x + self.mlp(self.process_attn(x, mask)).contiguous()

        return out


class lSTM_LAYER(nn.Module):
    def __init__(
        self,
        config: LayerConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype

        self.lstm = nn.LSTM(
            config.embed_dim,
            config.embed_dim,
            dropout=config.dropout,
            batch_first=True,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor):
        out, _ = self.lstm(x)
        return out


@component
class Abstractor(nn.Module, Model[ModelConfig, BatchInput, Output]):
    """Multi level sequence abstraction"""

    config = ModelConfig

    def __init__(self, config: ModelConfig, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.abstracted_seq_len = config.abstracted_seq_len
        self.config = config
        self.device = device
        self.embed = nn.Embedding(
            config.vocab_size, config.embed_dim, device=device, dtype=dtype
        )

        abstraction_scorers = []
        process_layers = []

        for i in range(config.n_abstractions):
            layer = MultiHeadSelfAttn(
                config.embed_dim,
                config.hidden_dim,
                config.abstraction_heads,
                config.process_heads,
                config.dropout,
                device=self.device,
                dtype=dtype,
            )
            abstraction_scorers.append(layer)

            for i in range(config.n_layers):
                layer_config = LayerConfig(
                    process_heads=config.process_heads,
                    embed_dim=config.embed_dim,
                    hidden_dim=config.hidden_dim,
                    dropout=config.dropout,
                )

                process_layers.append(
                    HAttnBlock(layer_config, device=device, dtype=dtype)
                )

        self.process_layers = nn.ModuleList(process_layers)
        self.abstraction_scorers = nn.ModuleList(abstraction_scorers)

        self.head_scorer = MultiHeadSelfAttn(
            config.embed_dim,
            config.hidden_dim,
            1,
            config.process_heads,
            config.dropout,
            device,
            dtype,
        )
        self.head_transform = nn.Linear(config.embed_dim, 1, device=device, dtype=dtype)

    def forward(self, batch: BatchInput) -> Output:

        embeddings = self.embed(batch.x)
        B, T, C = embeddings.shape

        mask = torch.ones(
            (B, self.abstracted_seq_len, self.abstracted_seq_len),
            device=self.device,
        )

        out = embeddings
        for i in range(self.config.n_abstractions):
            for j in range(self.config.n_layers):
                out = self.process_layers[i + j](out, batch.mask)
            
            scores = self.abstraction_scorers[i](out, batch.mask)

            _, inds = torch.topk(scores, k=min(T, self.abstracted_seq_len), dim=1)
            out = out.unsqueeze(-1).expand(B, -1, -1, self.config.abstraction_heads)
            out = out * scores.unsqueeze(-2)
            inds = inds.unsqueeze(-2).expand(B, -1, self.config.embed_dim, -1)
            out = torch.gather(out, dim=1, index=inds).squeeze(-1)
            out = torch.sum(out, dim=-1)
        scores = self.head_scorer(out, mask)
        dist_logits = self.head_transform(out)
        logits = torch.sum(dist_logits * scores, dim=1)

        probs = torch.nn.functional.sigmoid(logits)

        return Output(logits=logits, probs=probs)


if __name__ == "__main__":
    embed_dim = 4
    seq_len = 110
    config = ModelConfig(
        process_heads=1,
        n_layers=4,
        n_abstractions=2,
        abstraction_heads=4,
        abstracted_seq_len=100,
        vocab_size=200_000,
        embed_dim=embed_dim,
        hidden_dim=8,
        n_out=1,
        dropout=0.05,
    )
    model = Abstractor(
        config,
        torch.device("cpu"),
        torch.float32,
    )
    input = torch.randint(1, 20, (2, seq_len))
    mask = torch.ones_like(input)[:, None, :].expand(-1, seq_len, -1).float()
    output = model(input, mask)
    print(*output)
