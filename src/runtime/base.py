"""Runtime and embedder protocols for local/Modal execution backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from data.sources.base import DataSource

if TYPE_CHECKING:
    from data.pipeline.describe import DescribeJob
    from data.pipeline.engineer import EngineerJob
    from data.preprocess.pipeline import PreprocessJob
    from training.pipeline.eval import EvalJob
    from training.pipeline.train import TrainJob


@dataclass(frozen=True, kw_only=True)
class RunResult:
    """Outcome of a train/eval job dispatched through a ``Runtime``.

    For local runs ``status`` is ``"completed"`` and ``metrics`` carries the
    final summary.  For spawned remote jobs ``status`` is ``"spawned"`` and
    ``modal_function_call_id`` is the join key used to poll completion.
    """

    run_id: str
    status: str
    checkpoints: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    modal_function_call_id: str | None = None


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

    def run_train(self, job: TrainJob) -> RunResult: ...

    def run_eval(self, job: EvalJob) -> RunResult: ...
