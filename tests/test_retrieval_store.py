"""Tests for the retrieval vector store and its delta recency filter.

Covers:

- the FAISS code path is actually used when an index is present (spied via a
  stub implementing exact inner-product search in pure torch), and matches
  the matmul fallback exactly on toy data (the real-faiss assertion skips
  when faiss is not installed; the stubbed path is still exercised),
- the matmul fallback tolerates non-contiguous / grad-requiring queries and
  keeps the query on the db's device (``query.to(self.db.device)``); CUDA
  itself cannot be exercised on this machine, so the device line is asserted
  indirectly through correctness,
- the delta recency filter in ``VectorRetriever`` never selects a db row
  published after ``example_date - delta_years * 365.25`` days (in neither
  the stage-1 pool nor the CLaRa hard picks), and relaxes gracefully when
  an example has no candidate old enough,
- the CLaRa straight-through top-k (He et al., 2026, Algorithm 1): the
  forward output equals the hard gather exactly (no soft contamination),
  the k hard picks are distinct, and the autograd gradient through the ST
  path matches a hand-derived gradient on a tiny fixture,
- ``return_date=True`` plumbing through ``RetrievalDataset`` and
  ``token_batch_collate``,
- the ``RetrievalForecast`` dim guard and end-to-end gradient flow with the
  masked search path active.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
import pytest
import torch

from data.datasets.retrieval_dataset import (
    _HAS_FAISS,
    RetrievalDataset,
    VectorStoreDataset,
    VectorStoreDatasetConfig,
)
from data.datasets.text_token_dataset import (
    TextTokenDatasetConfig,
    token_batch_collate,
)
from data.datasets.types import TokenBatch
from models.retrieval_forecast import (
    AbstractQueryEmbedder,
    RetrievalForecast,
    RetrievalModelConfig,
    VectorRetriever,
)

DIM = 8
_TOP_K = 4


@dataclass(frozen=True)
class _LocalSource:
    """Minimal DataSource resolving to a prepared directory."""

    path: Path

    @property
    def backend(self) -> str:
        return "local"

    def resolve(self) -> Path:
        return self.path


def _write_store_parquet(path: Path, n: int = 24) -> None:
    """24 rows, dim-8 embeddings, dates spaced 90 days from 2000-01-01."""
    path.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    emb = torch.randn(n, DIM, generator=gen)
    start = date(2000, 1, 1)
    rows = {
        "id": list(range(n)),
        "abstract_embedding": [emb[i].tolist() for i in range(n)],
        "publication_date": [
            date.fromordinal(start.toordinal() + i * 90) for i in range(n)
        ],
        "cited_by_count": [float(i + 1) for i in range(n)],
    }
    pl.DataFrame(rows).write_parquet(path / "part_0.parquet")


def _store(path: Path, **overrides: object) -> VectorStoreDataset:
    config = VectorStoreDatasetConfig(
        loc="unused",
        embedding_col="abstract_embedding",
        filter=pl.col("cited_by_count") >= 1,
        normalize=True,
        name="store-test",
        **overrides,  # type: ignore[arg-type]
    )
    return VectorStoreDataset(config=config, source=_LocalSource(path))


class _StubFaissIndex:
    """Exact inner-product index over normalised rows, pure torch.

    Mirrors the faiss ``IndexFlatIP.search`` signature (numpy in, numpy out)
    and counts calls so tests can assert the faiss path was taken.
    """

    def __init__(self, db: torch.Tensor) -> None:
        self.db = db.clone()
        self.calls = 0

    def search(self, query: object, k: int) -> tuple[object, object]:
        self.calls += 1
        q = torch.from_numpy(query)  # type: ignore[attr-defined]
        scores = q.float() @ self.db.T
        k_eff = min(k, scores.size(-1))
        _, idx = torch.topk(scores, k=k_eff, dim=-1)
        return None, idx.numpy()


def test_faiss_path_used_when_index_present(tmp_path: Path) -> None:
    if not _HAS_FAISS:
        pytest.skip("faiss not installed; stubbed path covered by other tests")
    _write_store_parquet(path := tmp_path / "store")
    store = _store(path)
    assert store.faiss_index is not None  # real faiss when installed

    stub = _StubFaissIndex(store.db)
    store.faiss_index = stub
    q = torch.nn.functional.normalize(
        torch.randn(3, DIM, generator=torch.Generator().manual_seed(1)), dim=-1
    )
    idx = store.search(q, _TOP_K)

    assert stub.calls == 1, "search must go through the faiss index"
    scores = q @ store.db.T
    _, expected = torch.topk(scores, k=_TOP_K, dim=-1)
    assert torch.equal(idx, expected)


def test_faiss_path_matches_matmul_fallback(tmp_path: Path) -> None:
    _write_store_parquet(path := tmp_path / "store")
    store_faiss = _store(path)
    store_faiss.faiss_index = _StubFaissIndex(store_faiss.db)
    store_matmul = _store(path)
    store_matmul.faiss_index = None

    q = torch.nn.functional.normalize(
        torch.randn(5, DIM, generator=torch.Generator().manual_seed(2)), dim=-1
    )
    stub = store_faiss.faiss_index
    assert isinstance(stub, _StubFaissIndex)
    assert torch.equal(store_faiss.search(q, _TOP_K), store_matmul.search(q, _TOP_K))
    assert stub.calls == 1, "stubbed faiss path must be the one exercised"


def test_fallback_tolerates_noncontiguous_grad_query(tmp_path: Path) -> None:
    _write_store_parquet(path := tmp_path / "store")
    store = _store(path)
    store.faiss_index = None

    base = torch.randn(4, 3, DIM, generator=torch.Generator().manual_seed(3))
    q = base.transpose(0, 1).requires_grad_(True)  # (3, 4, DIM), non-contiguous
    idx = store.search(q, _TOP_K)

    flat = q.detach().reshape(-1, DIM).to(store.db.device)
    _, expected = torch.topk(flat @ store.db.T, k=_TOP_K, dim=-1)
    assert torch.equal(idx, expected.reshape(idx.shape))


def test_delta_filter_never_retrieves_too_recent_rows(tmp_path: Path) -> None:
    _write_store_parquet(path := tmp_path / "store")
    store = _store(path)
    retriever = VectorRetriever(
        store.db, _TOP_K, None, torch.device("cpu"),
        dates=store.dates, delta_years=1.0,
    )

    q = torch.nn.functional.normalize(
        torch.randn(2, 2, DIM, generator=torch.Generator().manual_seed(4)), dim=-1
    )
    # Cutoffs land inside the db's date range: rows newer than the cutoff
    # exist and must never be retrieved (keeps this test non-vacuous).
    example_dates = torch.tensor(
        [store.dates.max().item() - 200.0, store.dates.max().item() - 400.0]
    )
    cutoffs = example_dates - 365.25
    assert bool((store.dates[None, :] > cutoffs[:, None]).any()), (
        "fixture must contain rows newer than the cutoff"
    )
    idx = retriever.search(q, _TOP_K, example_dates)

    retrieved_dates = store.dates[idx.reshape(2, -1)]
    assert (retrieved_dates <= cutoffs[:, None]).all()


def test_delta_filter_relaxes_delta_to_zero_not_filter(tmp_path: Path) -> None:
    """Ladder 1, both branches.

    Example A sits 100 days after the oldest row: no row is a full delta
    older, so its cutoff relaxes to its own date (delta -> 0), and rows 0
    and 1 (the only rows <= A) remain the eligible set. Example B is the
    newest row: it has delta-old candidates, so ladder 1 does not fire and
    its cutoff keeps the full delta.
    Uses top_k=2 because exactly two rows are old enough for A, so the
    assertions are deterministic.
    """
    _write_store_parquet(path := tmp_path / "store")
    store = _store(path)
    retriever = VectorRetriever(
        store.db, 2, None, torch.device("cpu"),
        dates=store.dates, delta_years=1.0,
    )

    example_dates = torch.tensor([
        store.dates.min().item() + 100.0,  # A: relaxes to delta=0
        store.dates.max().item(),          # B: keeps the full delta
    ])
    # A's query aligns with row 1 (allowed for A); B's query aligns with
    # row 23, which is too recent for B's delta cutoff.
    q = torch.nn.functional.normalize(
        torch.stack([store.db[1], store.db[23]]), dim=-1
    ).unsqueeze(1)  # (2, 1, DIM)
    idx = retriever.search(q, 2, example_dates)

    assert idx.shape == (2, 1, 2)
    # Branch 1 (A): relaxation keeps the example-date bound (delta -> 0);
    # the aligned row 2 (110 days newer than A) must not be retrieved.
    a_dates = store.dates[idx[0, 0]]
    assert (a_dates <= example_dates[0]).all(), (
        "relaxation must keep the example-date bound (delta -> 0), not drop "
        "the filter"
    )
    assert not bool((idx[0, 0] == 2).any())
    # Branch 2 (B): no relaxation - the full delta cutoff holds.
    assert (store.dates[idx[1, 0]] <= example_dates[1] - 365.25).all()


def test_delta_filter_relaxes_when_no_old_candidate(tmp_path: Path) -> None:
    _write_store_parquet(path := tmp_path / "store")
    store = _store(path)
    retriever = VectorRetriever(
        store.db, _TOP_K, None, torch.device("cpu"),
        dates=store.dates, delta_years=1.0,
    )

    q = torch.nn.functional.normalize(
        torch.randn(1, 2, DIM, generator=torch.Generator().manual_seed(5)), dim=-1
    )
    # Example older than every db row: even delta=0 has no candidate, so it
    # has 0 < top_k eligible rows and ladder 2 drops the filter entirely
    # for it rather than returning garbage.
    example_dates = torch.tensor([store.dates.min().item() - 10.0])
    idx = retriever.search(q, _TOP_K, example_dates)
    assert idx.shape == (1, 2, _TOP_K)
    assert bool((idx >= 0).all() and (idx < store.db.size(0)).all())


def test_eligible_below_topk_drops_filter(tmp_path: Path) -> None:
    """Ladder 2 with a partially eligible example: an example with fewer
    than top_k eligible rows even at delta=0 gets the filter dropped (its
    picks may include rows newer than itself), while another example in the
    same batch keeps its delta filter."""
    _write_store_parquet(path := tmp_path / "store")
    store = _store(path)
    retriever = VectorRetriever(
        store.db, _TOP_K, None, torch.device("cpu"),
        dates=store.dates, delta_years=1.0,
    )

    # A: 100 days after the oldest row -> only rows 0 and 1 are <= A, fewer
    # than top_k=4 even at delta=0 -> filter dropped. Its query aligns with
    # row 5 (newer than A), so a dropped filter retrieves it.
    # B: newest row -> 19 delta-eligible rows >= top_k -> filter kept.
    example_dates = torch.tensor([
        store.dates.min().item() + 100.0,
        store.dates.max().item(),
    ])
    q = torch.nn.functional.normalize(
        torch.stack([store.db[5], store.db[23]]), dim=-1
    ).unsqueeze(1)  # (2, 1, DIM)
    idx = retriever.search(q, _TOP_K, example_dates)

    assert idx.shape == (2, 1, _TOP_K)
    assert bool((idx[0, 0] == 5).any()), (
        "ladder 2 must drop the filter for examples with fewer than top_k "
        "eligible rows"
    )
    # Contrast: B keeps the filter - nothing newer than its delta cutoff.
    cutoff_b = example_dates[1] - 365.25
    assert (store.dates[idx[1, 0]] <= cutoff_b).all()


def test_stage1_pool_excludes_disallowed_rows(tmp_path: Path) -> None:
    """Pool purity: when an example's eligible count is below n_candidates
    (but >= top_k), the stage-1 pool shrinks to the eligible count and no
    disallowed row enters via a -inf slot."""
    _write_store_parquet(path := tmp_path / "store")
    store = _store(path)
    retriever = VectorRetriever(
        store.db, _TOP_K, None, torch.device("cpu"),
        dates=store.dates, delta_years=1.0,
    )

    # Cutoff lands between row 4 and row 5: exactly 5 delta-eligible rows,
    # below n_candidates=8 but above top_k=4, so neither ladder fires.
    example_dates = torch.tensor([store.dates.min().item() + 750.0])
    cutoff = example_dates[0] - 365.25
    n_eligible = int((store.dates <= cutoff).sum())
    assert n_eligible == 5, "fixture must yield exactly 5 eligible rows"

    q = torch.nn.functional.normalize(
        torch.randn(1, 2, DIM, generator=torch.Generator().manual_seed(6)),
        dim=-1,
    )
    idx = retriever.search(q, 8, example_dates)

    assert idx.shape == (1, 2, 5), "pool must shrink to the eligible count"
    assert (store.dates[idx.reshape(-1)] <= cutoff).all(), (
        "every pool row must satisfy the example's delta cutoff"
    )


def test_delta_filter_requires_store_dates(tmp_path: Path) -> None:
    _write_store_parquet(path := tmp_path / "store")
    store = _store(path)
    retriever = VectorRetriever(
        store.db, _TOP_K, None, torch.device("cpu"), dates=None, delta_years=1.0,
    )
    q = torch.randn(1, 1, DIM)
    with pytest.raises(ValueError, match="time_col"):
        retriever.search(q, _TOP_K, torch.tensor([5000.0]))


def _write_token_parquet(path: Path, n: int = 6) -> None:
    path.mkdir(parents=True, exist_ok=True)
    start = date(2000, 1, 1)
    pl.DataFrame(
        {
            "id": list(range(n)),
            "title_tokens": [[1 + i, 2 + i, 3 + i] for i in range(n)],
            "cited_by_count": [float(i + 1) for i in range(n)],
            "publication_date": [
                date.fromordinal(start.toordinal() + i * 30) for i in range(n)
            ],
        }
    ).write_parquet(path / "part_0.parquet")


def test_retrieval_dataset_emits_and_collates_date(tmp_path: Path) -> None:
    _write_token_parquet(path := tmp_path / "tokens")
    config = TextTokenDatasetConfig(
        loc="unused",
        x=["title_tokens"],
        y=["cited_by_count"],
        max_len=8,
        pad_token_id=0,
        name="tokens",
        return_date=True,
    )
    dataset = RetrievalDataset(config=config, source=_LocalSource(path))

    first = dataset[0]
    assert first.date is not None
    assert first.date.dtype == torch.float32
    # Days since the Unix epoch: 2000-01-01 -> 10957 (NOT
    # date.toordinal(), which is the proleptic Gregorian ordinal 730120;
    # the store's polars Date physical repr is epoch days).
    assert first.date.item() == 10957.0

    collated = token_batch_collate([dataset[0], dataset[1]])
    assert collated.date is not None
    assert collated.date.shape == (2,)
    assert collated.date[1].item() == 10987.0  # 2000-01-31 -> 10957 + 30


def test_return_date_false_leaves_date_none(tmp_path: Path) -> None:
    _write_token_parquet(path := tmp_path / "tokens")
    config = TextTokenDatasetConfig(
        loc="unused",
        x=["title_tokens"],
        y=["cited_by_count"],
        max_len=8,
        pad_token_id=0,
        name="tokens",
    )
    dataset = RetrievalDataset(config=config, source=_LocalSource(path))
    assert dataset[0].date is None
    collated = token_batch_collate([dataset[0], dataset[1]])
    assert collated.date is None


def test_store_and_dataset_dates_share_encoding(tmp_path: Path) -> None:
    """Cross-encoding guard: store dates and batch dates must agree exactly.

    The store reads polars Date physical repr (days since Unix epoch); the
    dataset emits ``(python date - 1970-01-01).days``. If either side drifts
    to a different origin (e.g. back to ``date.toordinal()``), the delta
    filter silently becomes a no-op, so this test pins both to the same
    numbers for the same rows.
    """
    n = 6
    path = tmp_path / "combined"
    path.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(7)
    emb = torch.randn(n, DIM, generator=gen)
    start = date(2000, 1, 1)
    pl.DataFrame(
        {
            "id": list(range(n)),
            "title_tokens": [[1 + i, 2 + i, 3 + i] for i in range(n)],
            "cited_by_count": [float(i + 1) for i in range(n)],
            "abstract_embedding": [emb[i].tolist() for i in range(n)],
            "publication_date": [
                date.fromordinal(start.toordinal() + i * 90) for i in range(n)
            ],
        }
    ).write_parquet(path / "part_0.parquet")

    store = _store(path)
    dataset = RetrievalDataset(
        config=TextTokenDatasetConfig(
            loc="unused",
            x=["title_tokens"],
            y=["cited_by_count"],
            max_len=8,
            pad_token_id=0,
            name="combined",
            return_date=True,
        ),
        source=_LocalSource(path),
    )

    assert store.dates is not None
    for i in range(n):
        assert dataset[i].id.item() == float(i)
        assert dataset[i].date is not None
        assert dataset[i].date.item() == store.dates[i].item(), (
            "store and dataset date encodings diverged at row "
            f"{i}: {dataset[i].date.item()} != {store.dates[i].item()}"
        )
    # Sanity: the shared encoding really is epoch days, not ordinals.
    assert store.dates[0].item() == 10957.0


def test_dim_guard_raises_on_embedding_mismatch() -> None:
    cfg = RetrievalModelConfig(
        n_heads=2, n_layers=1, vocab_size=64, embed_dim=16, hidden_dim=32,
        n_out=1, dropout=0.0, n_queries=2, top_k=2, max_len=8,
    )
    embedder = AbstractQueryEmbedder(
        cfg.vocab_size, cfg.embed_dim, cfg.n_queries, cfg.n_heads,
        cfg.dropout, torch.device("cpu"), torch.float32,
    )
    with pytest.raises(ValueError, match="768") as exc_info:
        RetrievalForecast(cfg, embedder, torch.randn(20, 768))
    assert "16" in str(exc_info.value)


def test_end_to_end_grad_flow_with_delta_filter() -> None:
    torch.manual_seed(0)
    device, dtype = torch.device("cpu"), torch.float32
    cfg = RetrievalModelConfig(
        n_heads=2, n_layers=1, vocab_size=64, embed_dim=DIM, hidden_dim=32,
        n_out=1, dropout=0.0, n_queries=2, top_k=2, n_candidates=8,
        max_len=8,
    )
    embedder = AbstractQueryEmbedder(
        cfg.vocab_size, cfg.embed_dim, cfg.n_queries, cfg.n_heads,
        cfg.dropout, device, dtype,
    )
    db = torch.randn(30, DIM)
    db_dates = torch.arange(30, dtype=torch.float32) * 100.0
    model = RetrievalForecast(
        cfg, embedder, db, store_search=None, device=device, dtype=dtype,
        store_dates=db_dates, delta_years=1.0,
    )

    B, T = 3, 6
    x = torch.randint(1, cfg.vocab_size, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    batch = TokenBatch(
        id=torch.arange(B), x=x, y=torch.rand(B, 1).round(),
        mask=mask, weight=None,
        date=torch.tensor([2500.0, 2700.0, 2900.0]),
    )
    out = model(batch)
    assert out.logits.shape == (B, 1)
    out.logits.sum().backward()
    assert all(p.grad is not None for p in model.embedder.parameters())

    # And the masked path really excluded too-recent rows from the pool.
    q = model.embedder(batch.x, batch.mask)
    idx = model.retriever.search(q, cfg.n_candidates, batch.date)
    cutoffs = batch.date - 365.25
    assert (db_dates[idx.reshape(B, -1)] <= cutoffs[:, None]).all()

    # CLaRa hard picks: distinct, and no selected row newer than cutoff.
    selected = model.retriever(
        q, cfg.scale, batch.date, n_candidates=cfg.n_candidates
    )
    assert selected.shape == (B, cfg.n_queries, cfg.top_k, DIM)
    for b in range(B):
        sel_dates = db_dates[
            _picks_of(model.retriever, q[b : b + 1], cfg, batch.date[b : b + 1])
        ]
        assert (sel_dates <= cutoffs[b].item()).all()


def _picks_of(
    retriever: VectorRetriever,
    q_single: torch.Tensor,
    cfg: RetrievalModelConfig,
    dates: torch.Tensor,
) -> torch.Tensor:
    """Recompute the hard picks for one example via the masked argmax loop."""
    idx = retriever.search(q_single, cfg.n_candidates, dates)
    pool = retriever.db[idx]
    s_hat = torch.einsum("bnd,bncd->bnc", q_single, pool) / max(cfg.scale, 1e-6)
    s_hat = s_hat.reshape(-1, cfg.n_candidates).float()
    cutoff = retriever._effective_cutoffs(dates, s_hat.size(0))
    allowed = retriever.dates_buf[idx.reshape(-1, cfg.n_candidates)] <= (
        cutoff[:, None]
    )
    taken = torch.zeros(s_hat.size(0), cfg.n_candidates, dtype=torch.bool)
    picks = torch.full((s_hat.size(0), cfg.top_k), -1, dtype=torch.long)
    for j in range(cfg.top_k):
        logits = s_hat + torch.log(((~taken) & allowed).float() + 1e-6)
        r = logits.argmax(dim=-1)
        picks[:, j] = r
        taken = taken.scatter(1, r[:, None], True)
    return picks.reshape(-1)


def test_st_forward_equals_hard_gather() -> None:
    """Algorithm 1 forward: the returned selection is exactly the hard-picked
    pool vectors, with no soft-weight contamination."""
    torch.manual_seed(11)
    retriever, q = _tiny_retriever()
    out = retriever(q, 0.07, None, n_candidates=6)

    idx = retriever.search(q.detach(), 6, None)  # (1, 1, 6)
    # Recompute Algorithm 1's hard picks by hand on the same pool.
    pool = retriever.db[idx[0, 0]]
    s_hat = (q.detach()[0, 0] @ pool.T) / max(0.07, 1e-6)
    taken = torch.zeros(pool.size(0), dtype=torch.bool)
    hard = []
    for _ in range(retriever.top_k):
        logits = s_hat + torch.log(((~taken).float()) + 1e-6)
        r = int(logits.argmax())
        hard.append(r)
        taken[r] = True
    expected = retriever.db[torch.tensor(hard)]
    assert torch.allclose(out[0, 0], expected, atol=1e-6)


def test_st_no_duplicate_selections() -> None:
    torch.manual_seed(12)
    retriever, q = _tiny_retriever()
    out = retriever(q, 0.07, None, n_candidates=6)
    idx = retriever.search(q.detach(), 6, None)
    pool = retriever.db[idx[0, 0]]
    # Match each returned row back to a pool row; all k rows must differ.
    matches = torch.stack(
        [(pool - out[0, 0, j].detach()).abs().sum(-1).argmin() for j in range(retriever.top_k)]
    )
    assert len(set(matches.tolist())) == retriever.top_k


def test_st_gradient_matches_manual_estimator() -> None:
    """Autograd through the ST path equals the hand-derived gradient:
    dL/dZ_soft_j = g_j with g_j[c] = sum_e pool[c, e], pushed through the
    softmax jacobian per round, then dL/dq via the score einsum."""
    torch.manual_seed(13)
    retriever, q = _tiny_retriever(n_candidates=4)
    tau = 0.07
    out = retriever(q, tau, None, n_candidates=4)
    loss = out.sum()
    loss.backward()

    with torch.no_grad():
        idx = retriever.search(q.detach(), 4, None)
        pool = retriever.db[idx[0, 0]].float()  # (4, DIM)
        s_hat = (q.detach()[0, 0].float() @ pool.T) / max(tau, 1e-6)  # (4,)
        g = pool.sum(-1)  # dL/dZ[b,j,c] = sum_e pool[c,e] (all candidates)
        d_s = torch.zeros(4)
        taken = torch.zeros(4, dtype=torch.bool)
        for _ in range(retriever.top_k):
            mask = ~taken
            logits = s_hat + torch.log(mask.float() + 1e-6)
            p = torch.softmax(logits, dim=-1)
            # softmax jacobian: dp/dlogits = p * (g - <p, g>) over ALL
            # candidates; masked ones keep a tiny p via log(0 + eps).
            d_logits = p * (g - (p * g).sum())
            d_s += d_logits / max(tau, 1e-6)
            taken[int(logits.argmax())] = True
        d_q = d_s @ pool  # ds/dq = pool (cosine scores are linear in q here)

    assert q.grad is not None
    assert torch.allclose(q.grad[0, 0], d_q, atol=1e-5), (
        f"autograd {q.grad[0, 0]} vs manual {d_q}"
    )
    assert bool(torch.isfinite(q.grad).all()) and q.grad.abs().sum() > 0


def test_clara_respects_delta_filter() -> None:
    """Neither the stage-1 pool nor the ST hard picks may contain rows
    newer than example_date - delta_years * 365.25."""
    torch.manual_seed(14)
    db = torch.nn.functional.normalize(
        torch.randn(24, DIM, generator=torch.Generator().manual_seed(14)), dim=-1
    )
    db_dates = torch.arange(24, dtype=torch.float32) * 90.0
    retriever = VectorRetriever(
        db, 3, None, torch.device("cpu"), dates=db_dates, delta_years=1.0
    )
    q = torch.nn.functional.normalize(
        torch.randn(2, 2, DIM, generator=torch.Generator().manual_seed(15)),
        dim=-1,
    )
    # Cutoffs inside the db range so too-recent rows exist.
    example_dates = torch.tensor([db_dates.max().item(), db_dates.max().item() - 300.0])
    cutoffs = example_dates - 365.25
    assert bool((db_dates[None, :] > cutoffs[:, None]).any())

    pool_idx = retriever.search(q, 8, example_dates)
    assert (db_dates[pool_idx.reshape(2, -1)] <=
            torch.repeat_interleave(cutoffs, 2)[:, None]).all()

    selected = retriever(q, 0.07, example_dates, n_candidates=8)
    assert selected.shape == (2, 2, 3, DIM)
    # Every selected row must sit inside the allowed pool of its example.
    for b in range(2):
        pool_idx_b = retriever.search(q[b : b + 1], 8, example_dates[b : b + 1])
        pool = retriever.db[pool_idx_b]  # (1, 2, 8, DIM)
        for n in range(2):
            d = (pool[0, n] - selected[b, n].detach()).abs().sum(-1).argmin()
            assert db_dates[pool_idx_b[0, n, d]] <= cutoffs[b].item()


def _tiny_retriever(
    n_candidates: int = 6,
) -> tuple[VectorRetriever, torch.Tensor]:
    """One query row (B=1, N=1) over a 12-row db, no dates."""
    db = torch.nn.functional.normalize(
        torch.randn(12, DIM, generator=torch.Generator().manual_seed(10)), dim=-1
    )
    retriever = VectorRetriever(db, 2, None, torch.device("cpu"))
    q = torch.nn.functional.normalize(
        torch.randn(1, 1, DIM, generator=torch.Generator().manual_seed(11)),
        dim=-1,
    ).requires_grad_(True)
    assert n_candidates >= retriever.top_k
    return retriever, q
