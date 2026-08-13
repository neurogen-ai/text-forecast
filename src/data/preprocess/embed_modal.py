"""Client-side wrapper for the Modal GPU embedder."""

from __future__ import annotations

from logging import getLogger
from typing import Any

from .embed import TextEmbedder

logger = getLogger(__name__)


class ModalEmbedder:
    """TextEmbedder that delegates encoding to a Modal GPU class.

    The output dimension is supplied at construction time so that the local
    Polars schema can be built without loading the model client-side.
    """

    def __init__(
        self,
        model_name: str,
        output_dim: int,
        batch_size: int,
        gpu_cls: Any,
    ) -> None:
        self.model_name = model_name
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._gpu = gpu_cls(model_name=model_name)
        logger.debug(
            "ModalEmbedder.__init__: model_name=%r output_dim=%d batch_size=%d",
            model_name,
            output_dim,
            batch_size,
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._batch_size <= 0 or len(texts) <= self._batch_size:
            logger.debug(
                "ModalEmbedder.encode: sending single batch of %d texts",
                len(texts),
            )
            return self._gpu.encode.remote(texts)

        logger.debug(
            "ModalEmbedder.encode: splitting %d texts into chunks of %d",
            len(texts),
            self._batch_size,
        )
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            chunk = texts[i : i + self._batch_size]
            logger.debug(
                "ModalEmbedder.encode: sending chunk %d-%d (%d texts)",
                i,
                min(i + self._batch_size, len(texts)),
                len(chunk),
            )
            results.extend(self._gpu.encode.remote(chunk))
        logger.debug(
            "ModalEmbedder.encode: collected %d embeddings",
            len(results),
        )
        return results
