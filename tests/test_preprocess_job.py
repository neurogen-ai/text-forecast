"""Serialisation + behaviour-preservation tests for PreprocessJob (plan 2.1 T3).

Primary coverage is the pickle round-trip: Modal dispatches jobs via
``fn.remote(job)``, i.e. pickle — that is the boundary the job must survive.
``to_dict``/``from_dict`` is secondary (metadata/debug use).
"""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest

from data.preprocess.pipeline import PreprocessJob, run_preprocess_pipeline
from data.preprocess.steps import (
    CleanStep,
    DropNaStep,
    EmbedStep,
    LengthTrim,
    TokeniseStep,
)
from data.sources.local import LocalDataSource


class _StubEmbedder:
    output_dim: int = 4

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]


class _StubRuntime:
    supported_source_backends: set[str] = {"local"}

    def get_embedder(self, key: str, **kwargs: Any) -> _StubEmbedder:
        return _StubEmbedder()


@pytest.fixture
def sources(tmp_path) -> tuple[LocalDataSource, LocalDataSource]:
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()
    pl.DataFrame(
        {
            "text": [
                "a perfectly valid sentence",
                "http spam link",  # nulled by drop_quality
                None,
                "another good long sentence",
            ],
            "language": ["en", "en", "en", "fr"],
            "is_license_safe": [True, True, True, True],
            "publication_date": [datetime(2020, 1, 1)] * 4,
            "field_id": [1, 2, 3, 4],
            "type": ["article"] * 4,
        }
    ).write_parquet(src_dir / "part_0.parquet")
    origin = LocalDataSource(base_dir=tmp_path, name="src")
    destination = LocalDataSource(base_dir=tmp_path, name="dst")
    return origin, destination


def _full_job(
    origin: LocalDataSource, destination: LocalDataSource
) -> PreprocessJob:
    """Fully-populated job covering all four step types."""
    return PreprocessJob(
        origin=origin,
        destination=destination,
        steps=(
            CleanStep(
                col="text",
                lowercase=True,
                trim=LengthTrim(min_chars=5),
            ),
            DropNaStep(cols=("text",)),
            TokeniseStep(tokeniser="tok.json", cols=("text",)),
            EmbedStep(
                embedder_key="stub",
                embedder_kwargs={"batch_size": 7},
                cols=("text",),
            ),
        ),
        n_partitions=2,
        rows_per_part=0,
        compression_level=3,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2021, 1, 1),
        field_id=[1, 2, 3],
        languages=["en"],
        types=["article"],
        filt_license=True,
        replace_non_permissive_cols=["text"],
        dry_run=False,
        max_threads=4,
        runtime_name="local",
        params={"experiment": "t3"},
    )


def test_pickle_roundtrip_full_job(sources) -> None:
    """Modal pickles jobs across fn.remote(job); job must survive exactly."""
    origin, destination = sources
    job = _full_job(origin, destination)
    restored = pickle.loads(pickle.dumps(job))
    assert restored == job


def test_to_dict_from_dict_roundtrip_full_job(sources) -> None:
    """Secondary serialisation path; all four step types present."""
    origin, destination = sources
    job = _full_job(origin, destination)
    d = job.to_dict()
    # DataSource fields intentionally stay as objects (not JSON primitives).
    assert isinstance(d["origin"], LocalDataSource)
    tags = [s["tag"] for s in d["steps"]]
    assert tags == ["clean", "drop_na", "tokenise", "embed"]
    restored = PreprocessJob.from_dict(d)
    assert restored == job
    # Nested dataclass (LengthTrim) must be rebuilt, not stay a dict.
    clean = next(s for s in restored.steps if isinstance(s, CleanStep))
    assert isinstance(clean.trim, LengthTrim)


def test_dispatch_order_is_respected(sources, tmp_path) -> None:
    """Multi-step tuples run in order: CleanStep(drop_quality) -> DropNa
    removes the quality-nulled row entirely (2 rows), whereas DropNa ->
    CleanStep(drop_quality) leaves it behind as a null (3 rows). Different
    heights prove execution order is honoured."""
    origin, destination = sources

    def run(steps: tuple) -> pl.DataFrame:
        dst = LocalDataSource(base_dir=tmp_path, name=f"dst_{steps[0].tag}")
        job = PreprocessJob(origin=origin, destination=dst, steps=steps)
        result = run_preprocess_pipeline(job, _StubRuntime())  # type: ignore[arg-type]
        return pl.read_parquet(result.resolve() / "part_0.parquet")

    clean_then_drop = run(
        (
            CleanStep(col="text"),  # nulls the "http ..." row
            DropNaStep(cols=("text",)),
        )
    )
    drop_then_clean = run(
        (
            DropNaStep(cols=("text",)),  # only removes the pre-existing null
            CleanStep(col="text"),  # now nulls the "http ..." row
        )
    )
    assert clean_then_drop.height == 2
    assert clean_then_drop["text"].null_count() == 0
    assert drop_then_clean.height == 3
    assert drop_then_clean["text"].null_count() == 1


def test_zero_length_steps(sources) -> None:
    """steps=() runs through and writes output unchanged apart from
    partitioning (global filters still apply)."""
    origin, destination = sources
    job = PreprocessJob(origin=origin, destination=destination, steps=())
    result = run_preprocess_pipeline(job, _StubRuntime())  # type: ignore[arg-type]
    out = result.resolve() / "part_0.parquet"
    assert out.exists()
    df = pl.read_parquet(out)
    # No global filters set and no steps: output identical to input apart
    # from partitioning.
    assert df.height == 4
    assert sorted(
        t for t in df["text"].to_list() if t is not None
    ) == [
        "a perfectly valid sentence",
        "another good long sentence",
        "http spam link",
    ]


# ---------------------------------------------------------------------------
# Behaviour-preservation golden-ish test
# ---------------------------------------------------------------------------

N_ROWS = 50


def _synthetic_rows() -> list[dict[str, Any]]:
    rows = []
    for i in range(N_ROWS):
        rows.append(
            {
                "text": "alpha beta gamma delta epsilon zeta " * (1 + i % 3)
                + str(i),
                "language": ["en", "fr"][i % 2],
                "is_license_safe": i % 5 != 0,
                "publication_date": datetime(2020, 1, 1) + timedelta(days=i),
                "field_id": i % 3,
                "type": ["article", "report"][i % 2],
            }
        )
    return rows


def test_global_filters_and_clean_trim(tmp_path) -> None:
    """Global filters + one CleanStep(min_chars trim) behave as specified.

    Intentional deltas vs the old folded-min semantics
    (max(mean - 3*sigma, min_len)):
      * ``min_chars`` is now a hard minimum — rows below it are always
        dropped, there is no statistical folding back down/up.
      * There is no lower sigma bound at all; ``max_sigma`` only cuts the
        upper tail.
    The expected survivor set is computed independently in plain Python.
    """
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()
    rows = _synthetic_rows()
    pl.DataFrame(rows).write_parquet(src_dir / "part_0.parquet")

    start = datetime(2020, 1, 3)
    end = datetime(2020, 2, 10)
    min_chars = 60

    job = PreprocessJob(
        origin=LocalDataSource(base_dir=tmp_path, name="src"),
        destination=LocalDataSource(base_dir=tmp_path, name="dst"),
        steps=(CleanStep(col="text", trim=LengthTrim(min_chars=min_chars)),),
        start_date=start,
        end_date=end,
        field_id=[0, 1],
        languages=["en"],
        types=["article"],
        filt_license=True,
    )
    result = run_preprocess_pipeline(job, _StubRuntime())  # type: ignore[arg-type]
    out_df = pl.read_parquet(result.resolve() / "part_0.parquet")

    expected_texts = [
        r["text"]
        for r in rows
        if r["is_license_safe"]
        and r["publication_date"] >= start
        and r["publication_date"] < end
        and r["field_id"] in (0, 1)
        and r["language"] == "en"
        and r["type"] == "article"
        and len(r["text"]) >= min_chars
    ]
    assert sorted(out_df["text"].to_list()) == sorted(expected_texts)
    assert out_df.height == len(expected_texts) > 0
