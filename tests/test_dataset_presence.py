"""Missing staged datasets fail fast naming the resolved path (2.0 §10.4)."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from data.datasets.text_token_dataset import TextTokenDataset, TextTokenDatasetConfig
from data.sources import DataSource


@dataclass(frozen=True)
class _MissingSource:
    """DataSource whose resolved path does not exist (empty Modal volume)."""

    path: Path

    @property
    def backend(self) -> str:
        return "local"

    def resolve(self) -> Path:
        return self.path


def test_missing_dataset_raises_with_resolved_path(tmp_path: Path) -> None:
    missing = tmp_path / "staged" / "my-dataset"
    config = TextTokenDatasetConfig(
        loc="unused",
        x=["title_tokens"],
        y=["citations"],
        max_len=8,
        pad_token_id=0,
        name="my-dataset",
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        TextTokenDataset(config=config, source=_MissingSource(path=missing))
    message = str(exc_info.value)
    assert str(missing) in message
    assert "my-dataset" in message
    assert "text-forecast preprocess" in message
