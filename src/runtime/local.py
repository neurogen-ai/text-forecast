"""Local workstation runtime implementation."""

from __future__ import annotations

from typing import Any

from data.preprocess.embed import get_embedder
from data.preprocess.pipeline import PreprocessJob, run_preprocess_pipeline
from data.sources.base import DataSource

from .base import Runtime, TextEmbedder


class LocalRuntime:
    """Local runtime: local filesystem sources and HuggingFace embedders."""

    supported_source_backends = {"local"}

    def get_embedder(self, key: str, **kwargs: Any) -> TextEmbedder:
        return get_embedder(key, **kwargs)

    def run_preprocess(self, job: PreprocessJob) -> DataSource:
        return run_preprocess_pipeline(job, self)


Runtime.register(LocalRuntime)
