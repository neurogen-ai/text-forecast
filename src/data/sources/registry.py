from __future__ import annotations

from typing import Any

from .base import SourceBackend


_SOURCE_BACKENDS: dict[str, type[SourceBackend]] = {}


def register_source_backend(name: str):
    """Decorator that registers a :class:`SourceBackend` implementation."""

    def decorator(cls: type[SourceBackend]) -> type[SourceBackend]:
        _SOURCE_BACKENDS[name] = cls
        return cls

    return decorator


def build_source_backend(name: str, config: dict[str, Any]) -> SourceBackend:
    """Instantiate the source backend registered under ``name``."""
    if name not in _SOURCE_BACKENDS:
        raise ValueError(
            f"Unknown source backend {name!r}. "
            f"Available: {list(_SOURCE_BACKENDS)}"
        )
    return _SOURCE_BACKENDS[name](config)


def available_source_backends() -> list[str]:
    return list(_SOURCE_BACKENDS)
