"""Fail-fast existence check for staged datasets.

A missing staged dataset must surface as a clear error naming the resolved
path — not as an opaque polars scan error deep inside training. This matters
most inside a Modal container, where the staged volume may simply be empty.
"""

from __future__ import annotations

from pathlib import Path

from data.sources import DataSource


def assert_dataset_present(source: DataSource, name: str) -> Path:
    """Resolve ``source`` and raise if the dataset directory is missing."""
    path = source.resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset {name!r} not found at resolved path {path} "
            f"(backend={source.backend}). Produce it with 'citef preprocess' "
            "or stage it onto the volume before running."
        )
    return path
