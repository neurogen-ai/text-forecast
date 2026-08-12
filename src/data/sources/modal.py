from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import DataSource, SourceBackend
from .registry import register_source_backend


@dataclass(frozen=True)
class ModalDataSource:
    """Modal volume dataset located at ``/modal/<volume>/<name>``."""

    volume: str
    name: str

    @property
    def backend(self) -> str:
        return "modal"

    def resolve(self) -> Path:
        return Path("/modal") / self.volume / self.name


@register_source_backend("modal")
class ModalSourceBackend:
    """Modal volume source backend."""

    name = "modal"

    def __init__(self, config: dict[str, Any]) -> None:
        self.volume = config["volume"]

    def get_source(self, name: str) -> DataSource:
        return ModalDataSource(volume=self.volume, name=name)
