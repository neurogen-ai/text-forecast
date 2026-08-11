"""Local workstation runtime implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data.preprocess.embed import get_embedder
from data.sources import LocalStagedSource
from data.sources.base import DataSource

from .base import Runtime, TextEmbedder


class LocalRuntime:
    """Local runtime: local filesystem sources and HuggingFace embedders."""

    def get_source(self, root: str | Path, name: str) -> DataSource:
        return LocalStagedSource(path=Path(root), name=name)

    def get_embedder(self, key: str, **kwargs: Any) -> TextEmbedder:
        return get_embedder(key, **kwargs)

    def maybe_download(self, source: DataSource) -> Path:
        return source.resolve()

    def maybe_upload(self, source: DataSource, local_path: Path) -> None:
        return None


Runtime.register(LocalRuntime)
