"""Runtime and embedder protocols for local/Modal execution backends."""

from __future__ import annotations

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
    """Execution backend: embedders and remote job dispatch."""

    @property
    def supported_source_backends(self) -> set[str]: ...

    def get_embedder(self, key: str, **kwargs: Any) -> TextEmbedder: ...

    def run_preprocess(self, job: "PreprocessJob") -> DataSource: ...
