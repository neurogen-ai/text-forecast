"""Client-side wrapper for the Modal GPU embedder."""

from __future__ import annotations

from typing import Any

from .embed import TextEmbedder


class ModalEmbedder:
    """TextEmbedder that delegates encoding to a Modal GPU class.

    The output dimension is supplied at construction time so that the local
    Polars schema can be built without loading the model client-side.
    """

    def __init__(self, model_name: str, output_dim: int, gpu_cls: Any) -> None:
        self.model_name = model_name
        self._output_dim = output_dim
        self._gpu = gpu_cls()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._gpu.encode.remote(texts)
