"""Runtime and embedder protocols for local/Modal execution backends."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from data.sources.base import DataSource

if TYPE_CHECKING:
    from data.pipeline.describe import DescribeJob
    from data.pipeline.engineer import EngineerJob
    from data.preprocess.pipeline import PreprocessJob


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

    def get_source(self, root: str | Path, name: str) -> DataSource: ...

    def get_embedder(self, key: str, **kwargs: Any) -> TextEmbedder: ...

    def run_preprocess(self, job: PreprocessJob) -> DataSource: ...

    def run_describe(self, job: DescribeJob) -> None: ...

    def run_engineer(self, job: EngineerJob) -> DataSource: ...
