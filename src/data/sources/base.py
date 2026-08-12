from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataSource(Protocol):
    """Backend-specific handle that resolves to a path usable by the local process.

    For local files this is simply ``base_dir / name``. For Modal it is the
    in-container mount point. Callers that need to read should validate
    existence themselves; ``resolve`` only returns the location.
    """

    @property
    def backend(self) -> str: ...

    def resolve(self) -> Path: ...


@runtime_checkable
class SourceBackend(Protocol):
    """Factory for :class:`DataSource` instances of a particular backend."""

    name: str

    def __init__(self, config: dict[str, Any]) -> None: ...

    def get_source(self, name: str) -> DataSource: ...
