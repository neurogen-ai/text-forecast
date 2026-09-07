"""Compile probe for the v2.4.0 candidate-strategy topic (owner-run).

Diagnostic tooling, NOT a test — run it on the Strix Halo venue:

    python scripts/probe_candidate_strategy_compile.py [--mode max-autotune]

For each candidate strategy (nearest / random / mixed) it compiles one
RetrievalForecast training step and runs it over a few batches whose
pool width varies (the pre-existing k_pool shape churn), then reports
graph breaks and recompiles. It finishes with one bf16 compiled step.

Descope trigger this probe informs (v2.4.0 proposal, risk table): if the
Gumbel-perturbed topk graph-breaks or recompiles beyond a handful of
shapes, the padded-pool fallback is tried; if padding alone does not fix
it, random/mixed are cut. Read the printed numbers against the nearest
baseline row.
"""

from __future__ import annotations

import argparse

import torch

from models.retrieval_forecast import (
    AbstractQueryEmbedder,
    RetrievalForecast,
    RetrievalModelConfig,
)


def _build(strategy: str, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(0)
    cfg = RetrievalModelConfig(
        n_heads=2, n_layers=2, vocab_size=1000, embed_dim=64, hidden_dim=128,
        n_out=1, dropout=0.0, n_queries=4, top_k=4, n_candidates=32,
        max_len=64,
        candidate_strategy=strategy,  # type: ignore[arg-type]
    )
    embedder = AbstractQueryEmbedder(
        cfg.vocab_size, cfg.embed_dim, cfg.n_queries, cfg.n_heads,
        cfg.dropout, device, dtype,
    )
    db = torch.nn.functional.normalize(
        torch.randn(4096, cfg.embed_dim, generator=torch.Generator().manual_seed(1)),
        dim=-1,
    )
    db_dates = torch.arange(4096, dtype=torch.float32) * 10.0
    model = RetrievalForecast(
        cfg, embedder, db, store_search=None, device=device, dtype=dtype,
        store_dates=db_dates, delta_years=1.0,
    )
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return model, opt


def _batches(B: int, vocab: int, device: torch.device):
    """Batches whose per-example eligible counts differ -> varying k_pool."""
    for step in range(3):
        x = torch.randint(1, vocab, (B, 64), device=device)
        mask = torch.ones(B, 64, dtype=torch.bool, device=device)
        # Varying example dates across steps: the batch-minimum eligible
        # count (k_pool) changes shape, the churn this probe must measure.
        date = 4096.0 * 10.0 - 200.0 * (step + 1) + torch.arange(
            B, device=device
        ) * 5.0
        from data.datasets.types import TokenBatch

        yield TokenBatch(
            id=torch.arange(B, device=device),
            x=x, y=torch.rand(B, 1, device=device).round(),
            mask=mask, weight=None, date=date,
        )


def _probe(strategy: str, dtype: torch.dtype, mode: str, device: torch.device) -> dict:
    from torch._dynamo.utils import counters

    counters.clear()
    model, opt = _build(strategy, dtype, device)
    model.compile(mode=mode, fullgraph=False)
    for batch in _batches(4, 1000, device):
        out = model(batch)
        out.logits.sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return {
        "strategy": strategy,
        "dtype": str(dtype),
        # counters["graph_break"] maps break reason -> count;
        # counters["stats"]["unique_graphs"] counts recompiles.
        "graph_breaks": sum(counters["graph_break"].values()),
        "recompiles": counters["stats"]["unique_graphs"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="max-autotune")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"venue device: {device}, compile mode: {args.mode}")
    results = []
    for strategy in ("nearest", "random", "mixed"):
        results.append(_probe(strategy, torch.float32, args.mode, device))
    # One bf16 compiled step (compute-dtype topic rides on this venue).
    results.append(_probe("mixed", torch.bfloat16, args.mode, device))

    print("\nstrategy   dtype      graph_breaks  recompiles")
    for r in results:
        print(
            f"{r['strategy']:<10} {r['dtype']:<10} "
            f"{r['graph_breaks']:<13} {r['recompiles']}"
        )
    print(
        "\nDescope check: random/mixed graph_breaks and recompiles must stay\n"
        "within a handful of the nearest baseline row; otherwise try\n"
        "fixed-width padded pools before cutting non-nearest strategies."
    )


if __name__ == "__main__":
    main()
