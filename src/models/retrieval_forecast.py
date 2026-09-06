"""Retrieval-forecast model: embed, pool-search, ST top-k, cross-attend.

Pipeline per batch:

1. An embedder encodes the target paper's tokens into ``N`` latent query
   embeddings (learned latent queries cross-attend to the token sequence).
2. Each query embedding runs a no-grad stage-1 search over a fixed vector
   database (loaded from a precomputed parquet embedding column by
   ``VectorStoreDataset``) for its ``n_candidates`` nearest rows; the
   differentiable top-``top_k`` selection then runs over that candidate
   pool with the straight-through estimator from CLaRa (He et al., 2026,
   "CLaRa: Bridging Retrieval and Generation with Continuous Latent
   Reasoning", Algorithm 1): the forward pass consumes exactly the hard
   discrete top-k picks, while the backward pass uses the softmax gradient
   over the whole pool (``Z = Z_hard + (Z_soft - SG(Z_soft))``), so the
   embedder learns which candidates the downstream loss needs.
   When the vector store exposes publication dates and the batch carries
   per-example dates, stage-1 search is an exact masked top-k: rows
   published less than ``delta_years`` before the example are neither
   hard-picked nor given soft mass.
3. The ``N * top_k`` selected vectors form a retrieved sequence, padded
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

from typing import NamedTuple, Protocol

import logging

import torch
import torch.nn as nn
from data.datasets.types import TokenBatch
from pydantic import (
    BaseModel,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
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
    top_k: PositiveInt  # vectors selected per query (CLaRa k)
    n_candidates: PositiveInt = 32  # stage-1 pool size (CLaRa D; paper: 20)
    max_len: PositiveInt  # padded token length of the target sequence
    scale: PositiveFloat = 0.07  # CLaRa temperature tau (scores / max(tau, eps))
    rope_base: PositiveFloat = 10_000.0

    @model_validator(mode="after")
    def _check_candidates(self) -> "RetrievalModelConfig":
        if self.n_candidates < self.top_k:
            raise ValueError(
                f"n_candidates ({self.n_candidates}) must be >= top_k "
                f"({self.top_k}): the CLaRa ST loop selects top_k distinct "
                "rows from the candidate pool"
            )
        return self


class Output(NamedTuple):
    """Logits only. Probabilities are derived by consumers (sigmoid)."""

    logits: Tensor


class StoreSearch(Protocol):
    """Callable signature of ``VectorStoreDataset.search`` (no-grad top-k)."""

    def __call__(self, query: Tensor, top_k: int) -> Tensor: ...


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
    """Fixed vector database with CLaRa straight-through top-k selection.

    Stage 1 is a no-grad search for ``n_candidates`` pool rows (FAISS store
    path when no dates are given, exact masked matmul against this module's
    device-local ``db`` buffer otherwise). Stage 2 is the differentiable
    top-k straight-through estimator from CLaRa (He et al., 2026,
    Algorithm 1): per selection round the hard pick is the masked argmax of
    the scaled pool scores, the soft distribution is a softmax over the
    whole masked pool, previously picked rows are masked out of both, and
    the returned selection is ``Z = Z_hard + (Z_soft - SG(Z_soft))``. The
    forward value is exactly the hard-picked pool vectors (train matches
    inference); gradients reach the query embeddings through ``Z_soft``
    only. The database side is gradient-cancelled by construction (buffer).

    With per-example ``dates`` (and a store that loaded its own), database
    rows published later than ``example_date - delta_years * 365.25`` days
    are excluded from every round's mask. Examples with no candidate that
    old get ``delta`` relaxed to 0 for that example; if the example is
    older than every database row the mask is dropped for it entirely
    (warned once).
    """

    _warned_no_candidates = False

    def __init__(
        self,
        db: Tensor,
        top_k: int,
        store_search: StoreSearch | None,
        device: torch.device,
        dates: Tensor | None = None,
        delta_years: float = 0.0,
    ):
        super().__init__()
        self.top_k = top_k
        self._store_search = store_search
        self.register_buffer(
            "db", torch.nn.functional.normalize(db.float(), dim=-1).to(device),
            persistent=False,
        )
        self.dates_buf: Tensor | None
        if dates is not None:
            self.register_buffer(
                "dates_buf",
                torch.as_tensor(dates, dtype=torch.float32).reshape(-1).to(device),
                persistent=False,
            )
        self.delta_years = float(delta_years)

    def _effective_cutoffs(self, dates: Tensor, rows: int) -> Tensor:
        """Per-example date cutoffs with the relaxation ladder applied.

        ``dates`` is ``(B,)`` days-since-epoch example dates; ``rows`` is
        B*N (each example repeats for its N queries). Returns ``(rows,)``
        float32 cutoffs such that db rows with ``store_dates <= cutoff``
        are eligible. Ladder: delta -> 0 only for examples with no
        candidate a full delta older; the filter is dropped (cutoff = +inf)
        for examples with fewer than ``top_k`` eligible rows even at
        delta = 0 (warned once). Post-ladder, every example has at least
        ``top_k`` eligible rows, so no ST round can be fully masked.
        """
        store_dates = getattr(self, "dates_buf", None)
        if store_dates is None:
            raise ValueError(
                "Per-example date filtering requires a vector store built "
                "with time_col set (store dates were not provided)"
            )
        d = dates.detach().reshape(-1).float().to(self.db.device)
        d = d.repeat_interleave(rows // d.size(0))  # (B*N,) per query row
        cutoff = d - self.delta_years * 365.25
        allowed = store_dates[None, :] <= cutoff[:, None]  # (rows, M)
        if not bool(allowed.any(dim=1).all()):
            # Ladder 1: relax delta to 0 ONLY for the rows that lack a
            # candidate a full delta older (gated on the ORIGINAL cutoff);
            # rows that already have delta-old candidates keep their
            # original cutoff, so the recency filter is not weakened for
            # them.
            relaxed = store_dates[None, :] <= d[:, None]
            cutoff = torch.where(
                ~allowed.any(dim=1) & relaxed.any(dim=1), d, cutoff
            )
            allowed = store_dates[None, :] <= cutoff[:, None]
        if not bool((allowed.sum(dim=1) >= self.top_k).all()):
            # Ladder 2: fewer than top_k eligible rows even at delta=0 ->
            # the ST loop would hit a fully-masked round (argmax over an
            # all-masked row silently picks pool slot 0), so drop the
            # filter for those rows only.
            if not VectorRetriever._warned_no_candidates:
                logging.getLogger(__name__).warning(
                    "delta filter: some examples have fewer than top_k "
                    "candidate rows old enough; dropping the recency filter "
                    "for those examples"
                )
                VectorRetriever._warned_no_candidates = True
            cutoff = cutoff.masked_fill(
                allowed.sum(dim=1) < self.top_k, float("inf")
            )
            allowed = store_dates[None, :] <= cutoff[:, None]
        if not bool((allowed.sum(dim=1) >= self.top_k).all()):
            raise ValueError(
                f"vector db has fewer than top_k ({self.top_k}) rows "
                "eligible even without the recency filter; the CLaRa ST "
                "loop cannot select top_k distinct rows"
            )
        return cutoff

    @torch.no_grad()
    def search(self, q: Tensor, n_candidates: int, dates: Tensor | None = None) -> Tensor:
        """(B, N, D) -> (B, N, n_candidates) stage-1 pool row indices.

        ``dates`` is ``(B,)`` float32 days-since-epoch values, or None for
        unfiltered search (FAISS store path when available). With dates,
        the pool width shrinks to ``min(n_candidates, min eligible count)``
        so every returned row satisfies its example's cutoff (pool purity:
        no disallowed row can enter via a ``-inf`` slot). Without dates and
        with a db smaller than ``n_candidates``, a faiss index would return
        ``-1`` labels for the missing slots (which index-wrap to the last
        row), so that corner raises instead.
        """
        B, N, _ = q.shape
        if dates is None:
            if self.db.size(0) < n_candidates:
                raise ValueError(
                    f"vector db has {self.db.size(0)} rows but "
                    f"n_candidates is {n_candidates}; a faiss index would "
                    "return -1 labels for the missing slots (which "
                    "index-wrap to the last row) and the matmul fallback "
                    "would return a short pool"
                )
            if self._store_search is not None:
                idx = self._store_search(q.detach(), n_candidates)
                return idx.reshape(B, N, n_candidates).to(q.device)
            scores = q.detach().reshape(B * N, -1).float() @ self.db.T  # (B*N, M)
            k = min(n_candidates, scores.size(-1))
            _, idx = torch.topk(scores, k=k, dim=-1)
            return idx.reshape(B, N, k)

        flat = q.detach().reshape(B * N, -1).float()  # (B*N, D)
        cutoff = self._effective_cutoffs(dates, B * N)
        store_dates = getattr(self, "dates_buf")
        allowed = store_dates[None, :] <= cutoff[:, None]  # (B*N, M)
        # Pool purity: cap the pool at the smallest per-row eligible count
        # (>= top_k after the ladder) so a disallowed row can never fill a
        # ``-inf`` slot and later win an early ST round.
        k_pool = min(n_candidates, int(allowed.sum(dim=1).min().item()))
        scores = flat @ self.db.T  # (B*N, M)
        scores = scores.masked_fill(~allowed, float("-inf"))
        _, idx = torch.topk(scores, k=k_pool, dim=-1)
        return idx.reshape(B, N, k_pool)

    def forward(
        self,
        q: Tensor,
        scale: float,
        dates: Tensor | None = None,
        n_candidates: int | None = None,
    ) -> Tensor:
        """(B, N, D) -> selected vectors (B, N, top_k, D).

        CLaRa straight-through top-k over the stage-1 candidate pool
        (Algorithm 1 of He et al., 2026). Forward value: exactly the
        hard-picked pool vectors. Backward: softmax gradient over the
        masked pool, reaching ``q``.
        """
        B, N, _ = q.shape
        C = n_candidates if n_candidates is not None else self.top_k
        if C < self.top_k:
            raise ValueError(
                f"n_candidates ({C}) must be >= top_k ({self.top_k})"
            )
        # Algorithm 1, input: stage-1 pool over the (masked) scores.
        idx = self.search(q, C, dates)  # (B, N, pool), no grad
        C = idx.size(-1)  # min(n_candidates, db rows)
        if idx.size(-1) < self.top_k:
            raise ValueError(
                f"Stage-1 pool has {idx.size(-1)} rows but top_k is "
                f"{self.top_k}; the vector db is smaller than the model "
                "configuration assumes"
            )
        pool = self.db[idx].to(q.dtype)  # (B, N, C, D), constants
        # ~s = s / max(tau, 1e-6): cosine scores scaled by the temperature.
        s_hat = torch.einsum("bnd,bncd->bnc", q, pool) / max(scale, 1e-6)
        s_hat = s_hat.reshape(B * N, C).float()  # (B*N, C)
        pool_flat = pool.reshape(B * N, C, -1).float()

        # Eligibility mask for the pool rows under the same relaxed cutoffs
        # the stage-1 search used (per-example date filter; all-True when
        # no dates are given).
        if dates is None:
            allowed_pool = torch.ones_like(s_hat, dtype=torch.bool)
        else:
            store_dates = getattr(self, "dates_buf")
            cutoff = self._effective_cutoffs(dates, B * N)  # (B*N,)
            pool_dates = store_dates[idx.reshape(B * N, C)]  # (B*N, C)
            allowed_pool = pool_dates <= cutoff[:, None]

        # CLaRa loop, vectorised over the B*N query rows.
        eps = 1e-6  # Algorithm 1 log(mask + eps); value unspecified in paper
        BN = B * N
        z_hard = torch.zeros(BN, self.top_k, C, device=s_hat.device)
        z_soft = torch.zeros(BN, self.top_k, C, device=s_hat.device)
        taken = torch.zeros(BN, C, dtype=torch.bool, device=s_hat.device)
        for j in range(self.top_k):
            # (2) soft+hard mask: 1 - SG(taken), times the date eligibility.
            mask = (~taken) & allowed_pool  # (B*N, C)
            # Follows Algorithm 1: tau divides s only (s_hat). Main-text
            # eq 3.3 (tau outside the whole expression) is NOT equivalent:
            # there a masked entry loses unconditionally, here only via
            # log(eps).
            logits = s_hat + torch.log(mask.float() + eps)
            # (1) hard selection: argmax over unmasked candidates.
            r_j = logits.argmax(dim=-1)  # (B*N,)
            z_hard[:, j].scatter_(1, r_j[:, None], 1.0)
            # (2) soft selection: softmax over the whole masked pool.
            z_soft[:, j] = torch.softmax(logits, dim=-1)
            # (3) taken = min(taken + onehot(r_j), 1).
            taken = taken.scatter(1, r_j[:, None], True)
        # Straight-through: forward value is Z_hard; gradient flows via Z_soft.
        z = z_hard + (z_soft - z_soft.detach())

        # M^(k) = Z M: gather the hard-picked pool vectors in forward.
        selected = torch.einsum("bkc,bce->bke", z, pool_flat)
        return selected.to(q.dtype).reshape(B, N, self.top_k, -1)


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
        store_search: StoreSearch | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        store_dates: Tensor | None = None,
        delta_years: float = 0.0,
    ):
        super().__init__()
        device = torch.device(device)
        self.config = config
        if db.size(-1) != config.embed_dim:
            raise ValueError(
                f"Vector db dim {db.size(-1)} != model embed_dim "
                f"{config.embed_dim}; the precomputed embedding column and "
                "RetrievalModelConfig.embed_dim must match"
            )
        self.embedder = embedder
        self.retriever = VectorRetriever(
            db, config.top_k, store_search, device,
            dates=store_dates, delta_years=delta_years,
        )

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

        # Selected sequence: each of N query embeddings contributes top_k
        # hard-picked vectors (CLaRa ST forward), so the retrieved sequence
        # has length N * top_k.
        q = self.embedder(batch.x, batch.mask)  # (B, N, D)
        retrieved = self.retriever(
            q, cfg.scale, batch.date, n_candidates=cfg.n_candidates
        )
        B, N, K, D = retrieved.shape
        retrieved = retrieved.reshape(B, N * K, D)
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
        n_out=1, dropout=0.0, n_queries=4, top_k=3, n_candidates=10,
        max_len=10,
    )
    embedder = AbstractQueryEmbedder(
        cfg.vocab_size, cfg.embed_dim, cfg.n_queries, cfg.n_heads,
        cfg.dropout, device, dtype,
    )
    db = torch.randn(50, cfg.embed_dim)
    db_dates = torch.arange(50, dtype=torch.float32) * 100.0  # days since epoch
    model = RetrievalForecast(
        cfg, embedder, db, device=device, dtype=dtype,
        store_dates=db_dates, delta_years=1.0,
    )

    B, T = 4, 7
    x = torch.randint(1, cfg.vocab_size, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 5:] = False
    batch = TokenBatch(
        id=torch.arange(B), x=x, y=torch.rand(B, 1).round(),
        mask=mask, weight=None,
        date=torch.tensor([4000.0, 4500.0, 5000.0, 6000.0]),
    )
    out = model(batch)
    assert out.logits.shape == (B, 1), out.logits.shape
    out.logits.sum().backward()
    grads = [p.grad is not None for p in model.embedder.parameters()]
    assert all(grads), grads
    # Delta filter respected: no selected row newer than cutoff.
    idx = model.retriever.search(
        model.embedder(x, mask), cfg.n_candidates, batch.date
    ).reshape(B, -1)
    cutoffs = batch.date - 365.25
    assert (db_dates[idx] <= cutoffs[:, None]).all()
    # And gradient reaches the embedder through the ST path.
    out = model(batch)
    out.logits.sum().backward()
    lat = model.embedder.latents
    assert lat.grad is not None and bool(torch.isfinite(lat.grad).all())
    print("ok", out.logits.shape, "embedder grads:", all(grads))
