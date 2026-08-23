"""Local workstation runtime implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data.pipeline.describe import DescribeJob, run_describe_pipeline
from data.pipeline.engineer import EngineerJob, run_engineer_pipeline
from data.preprocess.embed import get_embedder
from data.preprocess.pipeline import PreprocessJob, run_preprocess_pipeline
from data.sources.base import DataSource
from data.sources.local import LocalDataSource

from .base import Runtime, TextEmbedder


class LocalRuntime:
    """Local runtime: local filesystem sources and HuggingFace embedders."""

    supported_source_backends = {"local"}

    def get_source(self, root: str | Path, name: str) -> DataSource:
        # No existence check: output sources do not exist until written.
        return LocalDataSource(base_dir=Path(root), name=name)

    def get_embedder(self, key: str, **kwargs: Any) -> TextEmbedder:
        return get_embedder(key, **kwargs)

    def run_preprocess(self, job: PreprocessJob) -> DataSource:
        return run_preprocess_pipeline(job, self)

    def run_describe(self, job: DescribeJob) -> None:
        run_describe_pipeline(job)

    def run_engineer(self, job: EngineerJob) -> DataSource:
        return run_engineer_pipeline(job)


Runtime.register(LocalRuntime)
