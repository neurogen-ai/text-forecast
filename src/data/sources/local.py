from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import DataSource, SourceBackend
from .registry import register_source_backend


@dataclass(frozen=True)
class LocalDataSource:
    """Local filesystem dataset located at ``base_dir / name``."""

    base_dir: Path
    name: str

    @property
    def backend(self) -> str:
        return "local"

    def resolve(self) -> Path:
        return self.base_dir / self.name


@register_source_backend("local")
class LocalSourceBackend:
    """Local filesystem source backend."""

    name = "local"

    def __init__(self, config: dict[str, Any]) -> None:
        self.base_dir = Path(config["base_dir"])
        if not self.base_dir.exists():
            raise FileNotFoundError(
                f"Local source base_dir does not exist: {self.base_dir}"
            )

    def get_source(self, name: str) -> DataSource:
        return LocalDataSource(base_dir=self.base_dir, name=name)
