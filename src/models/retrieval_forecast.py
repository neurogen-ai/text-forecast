"""Retrieval-forecast model: embed, search, softmax top-k, cross-attend.

Pipeline per batch:

1. An embedder encodes the target paper's tokens into ``N`` latent query
   embeddings (learned latent queries cross-attend to the token sequence).
2. Each query embedding searches a fixed vector database (loaded from a
   precomputed parquet embedding column by ``VectorStoreDataset``) for its
   ``top_k`` nearest vectors. The search itself runs under ``no_grad``; the
   *selection* is then re-scored differentiably: similarities between the
   query embeddings and the retrieved (gradient-cancelled) database vectors
   go through a softmax, so gradients reach the embedder and it can learn
   how effective the vector search is. This follows the soft top-k selection
   recipe from the February 2025 Appli paper on differentiable retrieval.
3. The ``N * top_k`` retrieved vectors form a retrieved sequence, padded
   alongside the target token sequence to the longer of the two lengths.
4. A predictor cross-attends from the target token sequence to the retrieved
   sequence, then runs RoPE self-attention layers and an attention-pooling
   head.

Positional encoding is RoPE (rotary embeddings) applied to queries and keys
in every attention layer; positions are shared between the two padded
sequences so cross-attention sees a common coordinate frame.

Target: binary threshold on citation count; output is logits only, consumed
by ``BinaryCrossEntropyLoss`` / ``ClassificationStrategy``.
"""

from typing import NamedTuple

import torch
import torch.nn as nn
from data.datasets.types import TokenBatch
from pydantic import BaseModel, PositiveFloat, PositiveInt
from torch import Tensor

from utils import component

from .protocols import Model


class RetrievalModelConfig(BaseModel):
    n_heads: int
    n_layers: PositiveInt
    vocab_size: PositiveInt
    embed_dim: PositiveInt
    hidden_dim: PositiveInt
    n_out: PositiveInt
    dropout: float
    n_queries: PositiveInt  # N: latent query embeddings per paper
    top_k: PositiveInt  # vectors retrieved per query
    max_len: PositiveInt  # padded token length of the target sequence
    scale: PositiveFloat = 0.07  # temperature for the softmax top-k
    rope_base: PositiveFloat = 10_000.0


class Output(NamedTuple):
    """Logits only. Probabilities are derived by consumers (sigmoid)."""

    logits: Tensor


class RotaryEmbedding(nn.Module):
    """Precomputed rotary tables for positions ``[0, max_len)``."""

    def __init__(
        self,
        head_dim: int,
        max_len: int,
        base: float,
        device: torch.device,
    ):
        super().__init__()
        self.max_len = max_len
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
                / head_dim
            )
        )
        t = torch.arange(max_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_len, head_dim // 2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[Tensor, Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate (B, H, T, head_dim) with (T, head_dim // 2) tables."""
    x1, x2 = x.chunk(2, dim=-1)
    c = cos.to(x.dtype)[None, None]
    s = sin.to(x.dtype)[None, None]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class RoPEAttention(nn.Module):
    """Multi-head attention with RoPE on queries and keys.

    With ``q_in is kv_in`` this is self-attention; otherwise the target
    sequence attends to another sequence (cross-attention). Both sequences
    are padded to the same length, so a single position table covers both.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float,
        rope: RotaryEmbedding,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, "Embed dim must split evenly across heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.rope = rope
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.kv_proj = nn.Linear(
            embed_dim, embed_dim * 2, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

    def forward(
        self, q_in: Tensor, kv_in: Tensor, kv_mask: Tensor
    ) -> Tensor:
        """(B, Tq, C), (B, Tk, C), (B, Tk) bool -> (B, Tq, C)."""
        B, Tq, _ = q_in.shape
        Tk = kv_in.size(1)
        q = self.q_proj(q_in).reshape(
            B, Tq, self.n_heads, self.head_dim
        ).transpose(1, 2)
        kv = self.kv_proj(kv_in).reshape(
            B, Tk, self.n_heads, 2 * self.head_dim
        )
        k, v = kv.chunk(2, dim=-1)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos_q, sin_q = self.rope(Tq)
        cos_k, sin_k = self.rope(Tk)
        q = apply_rope(q, cos_q, sin_q)
        k = apply_rope(k, cos_k, sin_k)

        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=kv_mask[:, None, None, :], dropout_p=self.dropout
        )
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.embed_dim)
        return self.out_proj(out)


class AbstractQueryEmbedder(nn.Module):
    """Token embedder producing N query embeddings per paper.

    N learned latent queries cross-attend to the token sequence (position
    free on the latent side, so no RoPE there); each paper's N embeddings
    retrieve ``top_k`` vectors from the database.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_queries: int,
        n_heads: int,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_queries = n_queries
        self.token_embed = nn.Embedding(
            vocab_size, embed_dim, device=device, dtype=dtype
        )
        self.latents = nn.Parameter(
            torch.randn(n_queries, embed_dim, device=device, dtype=dtype) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True,
            device=device, dtype=dtype,
        )
        self.norm = nn.RMSNorm(embed_dim, device=device, dtype=dtype)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """(B, T), (B, T) -> (B, N, D) normalised query embeddings."""
        emb = self.token_embed(x)
        latents = self.latents.unsqueeze(0).expand(x.size(0), -1, -1)
        attn_out, _ = self.cross_attn(
            latents, emb, emb, key_padding_mask=~mask, need_weights=False
        )
        q = self.norm(latents + attn_out)
        return torch.nn.functional.normalize(q, dim=-1)


class VectorRetriever(nn.Module):
    """Fixed vector database with differentiable softmax top-k selection.

    The FAISS/matmul search runs under ``no_grad`` (indices and raw distances
    carry no gradient). Selection is then re-scored by dot products between
    the live query embeddings and the retrieved database vectors, and a
    softmax turns those scores into weights. Gradients flow into the query
    embeddings only; the database side is gradient-cancelled by construction
    (it is a buffer).
    """

    def __init__(
        self, db: Tensor, top_k: int, store_search: object, device: torch.device
    ):
        super().__init__()
        self.top_k = top_k
        self._store_search = store_search
        self.register_buffer(
            "db", torch.nn.functional.normalize(db.float(), dim=-1).to(device),
            persistent=False,
        )

    @torch.no_grad()
    def search(self, q: Tensor) -> Tensor:
        """(B, N, D) -> (B, N, top_k) database row indices."""
        B, N, _ = q.shape
        if self._store_search is not None:
            idx = self._store_search(q.detach(), self.top_k)  # type: ignore[operator]
            return idx.reshape(B, N, self.top_k).to(q.device)
        scores = q.reshape(B * N, -1).float() @ self.db.T  # (B*N, M)
        k = min(self.top_k, scores.size(-1))
        _, idx = torch.topk(scores, k=k, dim=-1)
        return idx.reshape(B, N, self.top_k)

    def forward(self, q: Tensor, scale: float) -> tuple[Tensor, Tensor]:
        """(B, N, D) -> retrieved (B, N, top_k, D) and weights (B, N, top_k)."""
        idx = self.search(q)  # no grad through the search
        retrieved = self.db[idx].to(q.dtype)  # constant database vectors
        # Differentiable re-scoring: gradient reaches q, not the db.
        scores = torch.einsum("bnd,bnkd->bnk", q, retrieved) / scale
        weights = torch.softmax(scores, dim=-1)
        return retrieved, weights


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        n_heads: int,
        dropout: float,
        rope: RotaryEmbedding,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.attn = RoPEAttention(embed_dim, n_heads, dropout, rope, device, dtype)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn(x, x, mask)
        x = x + self.mlp(nn.functional.rms_norm(x, (x.size(-1),)))
        return x


class HeadScorer(nn.Module):
    """RoPE self-attention that emits one score per position."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float,
        rope: RotaryEmbedding,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.attn = RoPEAttention(embed_dim, n_heads, dropout, rope, device, dtype)
        self.out = nn.Linear(embed_dim, 1, device=device, dtype=dtype)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.out(self.attn(x, x, mask))


def pad_to_length(
    x: Tensor, mask: Tensor, target_len: int
) -> tuple[Tensor, Tensor]:
    """Zero-pad an embedding sequence (and its mask) to ``target_len``."""
    cur = x.size(1)
    if cur >= target_len:
        return x, mask
    pad = target_len - cur
    x = nn.functional.pad(x, (0, 0, 0, pad))
    mask = nn.functional.pad(mask, (0, pad), value=False)
    return x, mask


@component
class RetrievalForecast(nn.Module, Model[RetrievalModelConfig, TokenBatch, Output]):
    config = RetrievalModelConfig

    def __init__(
        self,
        config: RetrievalModelConfig,
        embedder: AbstractQueryEmbedder,
        db: Tensor,
        store_search: object | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        device = torch.device(device)
        self.config = config
        self.embedder = embedder
        self.retriever = VectorRetriever(db, config.top_k, store_search, device)

        # Shared position table across both padded sequences.
        rope_len = max(config.max_len, config.n_queries * config.top_k)
        rope = RotaryEmbedding(
            config.embed_dim // config.n_heads, rope_len, config.rope_base, device
        )

        self.cross_attn = RoPEAttention(
            config.embed_dim, config.n_heads, config.dropout, rope, device, dtype
        )
        self.cross_norm = nn.RMSNorm(config.embed_dim, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    config.embed_dim,
                    config.hidden_dim,
                    config.n_heads,
                    config.dropout,
                    rope,
                    device,
                    dtype,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.head_scorer = HeadScorer(
            config.embed_dim, config.n_heads, config.dropout, rope, device, dtype
        )
        self.head = nn.Linear(
            config.embed_dim, config.n_out, device=device, dtype=dtype
        )

    def forward(self, batch: TokenBatch) -> Output:
        cfg = self.config
        token_emb = self.embedder.token_embed(batch.x)

        # Retrieved sequence: each of N query embeddings contributes top_k
        # vectors, so the retrieved sequence has length N * top_k.
        q = self.embedder(batch.x, batch.mask)  # (B, N, D)
        retrieved, weights = self.retriever(q, cfg.scale)
        B, N, K, D = retrieved.shape
        retrieved = (retrieved * weights.unsqueeze(-1)).reshape(B, N * K, D)
        retrieved_mask = torch.ones(
            B, N * K, dtype=torch.bool, device=retrieved.device
        )

        # Pad the smaller sequence to the longer one (max_len vs N * top_k).
        seq_len = max(cfg.max_len, N * K)
        token_emb, batch_mask = pad_to_length(token_emb, batch.mask, seq_len)
        retrieved, retrieved_mask = pad_to_length(
            retrieved, retrieved_mask, seq_len
        )

        # Cross-attention from the retrieved sequence: the target tokens
        # attend to their retrieved neighbours.
        attn_out = self.cross_attn(token_emb, retrieved, retrieved_mask)
        out = self.cross_norm(token_emb + attn_out)

        for layer in self.layers:
            out = layer(out, batch_mask)

        # Attention pooling over real (unpadded) positions, as TransformerClass.
        mask = batch_mask.unsqueeze(-1)
        out = nn.functional.rms_norm(out, (out.size(-1),))
        raw_scores = self.head_scorer(out, batch_mask)
        masked_scores = raw_scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(masked_scores, dim=1)
        logits = torch.sum(self.head(out) * scores, dim=1)
        return Output(logits=logits)


if __name__ == "__main__":
    torch.manual_seed(0)
    device, dtype = torch.device("cpu"), torch.float32
    cfg = RetrievalModelConfig(
        n_heads=2, n_layers=2, vocab_size=100, embed_dim=16, hidden_dim=32,
        n_out=1, dropout=0.0, n_queries=4, top_k=3, max_len=10,
    )
    embedder = AbstractQueryEmbedder(
        cfg.vocab_size, cfg.embed_dim, cfg.n_queries, cfg.n_heads,
        cfg.dropout, device, dtype,
    )
    db = torch.randn(50, cfg.embed_dim)
    model = RetrievalForecast(cfg, embedder, db, device=device, dtype=dtype)

    B, T = 4, 7
    x = torch.randint(1, cfg.vocab_size, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 5:] = False
    batch = TokenBatch(
        id=torch.arange(B), x=x, y=torch.rand(B, 1).round(),
        mask=mask, weight=None,
    )
    out = model(batch)
    assert out.logits.shape == (B, 1), out.logits.shape
    out.logits.sum().backward()
    grads = [p.grad is not None for p in model.embedder.parameters()]
    assert all(grads), grads
    print("ok", out.logits.shape, "embedder grads:", all(grads))
