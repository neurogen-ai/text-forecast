"""Embedder protocol and key-based registry."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TextEmbedder(Protocol):
    """Pluggable text encoder: plain Python in/out, framework-agnostic call shape."""

    @property
    def output_dim(self) -> int: ...

    def encode(self, texts: list[str]) -> list[list[float]]: ...


_EMBEDDERS: dict[str, type[TextEmbedder]] = {}


def register_embedder(key: str):
    """Decorator that registers an embedder implementation under ``key``."""

    def decorator(cls: type[TextEmbedder]) -> type[TextEmbedder]:
        _EMBEDDERS[key] = cls
        return cls

    return decorator


def get_embedder(key: str, **kwargs: object) -> TextEmbedder:
    """Instantiate the embedder registered under ``key``."""
    if key not in _EMBEDDERS:
        raise ValueError(
            f"Unknown embedder {key!r}. Available: {list(_EMBEDDERS)}"
        )
    return _EMBEDDERS[key](**kwargs)


def available_embedders() -> list[str]:
    return list(_EMBEDDERS)
