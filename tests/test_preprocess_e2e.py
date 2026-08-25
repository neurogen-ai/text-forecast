"""End-to-end test for run_preprocess_pipeline (plan 2.1 T2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl
import pytest

from data.preprocess.pipeline import PreprocessJob, run_preprocess_pipeline
from data.preprocess.steps import CleanStep, DropNaStep, LengthTrim
from data.sources.local import LocalDataSource


class _StubEmbedder:
    output_dim: int = 4

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]


class _StubRuntime:
    """LocalRuntime-style runtime with a stubbed embedder."""

    supported_source_backends: set[str] = {"local"}

    def get_embedder(self, key: str, **kwargs: Any) -> _StubEmbedder:
        return _StubEmbedder()


@pytest.fixture
def fixture_dirs(tmp_path):
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()
    pl.DataFrame(
        {
            "text": ["hello world", "hi", None, "valid text here"],
            "language": ["en", "en", "en", "en"],
            "is_license_safe": [True, True, True, True],
            "publication_date": [datetime(2020, 1, 1)] * 4,
            "field_id": [1, 2, 3, 4],
            "type": ["article"] * 4,
        }
    ).write_parquet(src_dir / "part_0.parquet")
    return LocalDataSource(base_dir=tmp_path, name="src"), LocalDataSource(
        base_dir=tmp_path, name="dst"
    )


def test_run_preprocess_pipeline_clean_dropna(fixture_dirs) -> None:
    origin, destination = fixture_dirs
    job = PreprocessJob(
        origin=origin,
        destination=destination,
        steps=(
            CleanStep(col="text", trim=LengthTrim(min_chars=5)),
            DropNaStep(cols=("text",)),
        ),
        rows_per_part=10,
    )
    result = run_preprocess_pipeline(job, _StubRuntime())  # type: ignore[arg-type]
    out = result.resolve() / "part_0.parquet"
    assert out.exists()
    df = pl.read_parquet(out)
    assert df.height == 2
    assert df["text"].to_list() == ["hello world", "valid text here"]


def test_preprocess_job_roundtrip(fixture_dirs) -> None:
    origin, destination = fixture_dirs
    job = PreprocessJob(
        origin=origin,
        destination=destination,
        steps=(
            CleanStep(col="text", trim=LengthTrim(min_chars=5)),
            DropNaStep(cols=("text",)),
        ),
        rows_per_part=10,
    )
    restored = PreprocessJob.from_dict(job.to_dict())
    assert restored.steps == job.steps
