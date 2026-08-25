from typing import TYPE_CHECKING

from .base import DataSource, SourceBackend
from .local import LocalDataSource, LocalSourceBackend
from .modal import ModalDataSource, ModalSourceBackend
from .registry import (
    available_source_backends,
    build_source_backend,
    register_source_backend,
)

if TYPE_CHECKING:
    from config.env import Env


def build_default_source_backend(env: "Env") -> SourceBackend:
    """Build the backend named by ``env.source['default']`` with its config.

    Experiments call this inside ``build(runtime, env)`` so that dataset
    location stays machine-level (Env) while dataset identity stays in the
    experiment file.
    """
    name = env.source.get("default", "local")
    return build_source_backend(name, env.source.get(name, {}))

__all__ = [
    "DataSource",
    "SourceBackend",
    "LocalDataSource",
    "LocalSourceBackend",
    "ModalDataSource",
    "ModalSourceBackend",
    "available_source_backends",
    "build_source_backend",
    "build_default_source_backend",
    "register_source_backend",
]
