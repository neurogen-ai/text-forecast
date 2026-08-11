"""Modal volume-backed data source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import DataSource


@dataclass(frozen=True, kw_only=True)
class ModalVolumeSource:
    """In-container path resolver for a dataset stored on a Modal volume.

    The runtime mounts Modal volumes under ``/modal/<volume_label>``.  This
    source returns the fully-qualified in-container path so that Polars can
    read and write parquet files transparently, whether the job runs in the
    cloud or in a local Modal container.
    """

    volume_label: str
    path: str
    name: str

    def resolve(self) -> Path:
        return Path("/modal") / self.volume_label / self.path / self.name


DataSource.register(ModalVolumeSource)
