"""Runtime and embedder protocols for local/Modal execution backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from data.sources.base import DataSource


@runtime_checkable
class TextEmbedder(Protocol):
    """Pluggable text encoder: plain Python in/out, framework-agnostic call shape."""

    @property
    def output_dim(self) -> int: ...

    def encode(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Runtime(Protocol):
    """Execution backend: resolves paths, volumes, embedders, and remote jobs."""

    def get_source(self, root: str | Path, name: str) -> DataSource: ...

    def get_embedder(self, key: str, **kwargs: Any) -> TextEmbedder: ...

    def maybe_download(self, source: DataSource) -> Path: ...

    def maybe_upload(self, source: DataSource, local_path: Path) -> None: ...
