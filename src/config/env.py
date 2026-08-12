"""Environment / machine settings.

Typed :class:`Env` dataclass loaded from ``config/config.toml`` with optional
CLI overrides.  No module-level singletons or PEP 562 shims remain.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.get_root_dir import get_root_dir


@dataclass(frozen=True, kw_only=True)
class Env:
    """Machine-specific settings that are not experiment-relevant."""

    tracking_uri: str
    artifact_loc: Path
    runtime: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)


def _load_toml() -> dict[str, Any]:
    path = get_root_dir(markers=("pyproject.toml",)) / "config" / "config.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    """Merge ``updates`` into ``base`` recursively."""
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def load_env(
    overrides: dict[str, Any] | None = None,
    source_overrides: dict[str, Any] | None = None,
) -> Env:
    """Load ``[env]``, ``[source]`` and ``[runtime]`` from ``config/config.toml``.

    ``[source]`` defines the default data-source backend and its config. CLI
    flags can override ``tracking_uri``/``artifact_loc`` via ``overrides`` and
    source-backend options via ``source_overrides``.
    """
    overrides = overrides or {}
    data = _load_toml()
    env_data = data.get("env", {})

    source = data.get("source", {})
    source.setdefault("default", "local")
    source.setdefault("local", {})
    source["local"].setdefault("base_dir", "/tmp/data")
    if source_overrides:
        _deep_update(source, source_overrides)

    return Env(
        tracking_uri=overrides.get(
            "tracking_uri", env_data.get("tracking_uri", "http://127.0.0.1:5000")
        ),
        artifact_loc=Path(
            overrides.get(
                "artifact_loc", env_data.get("artifact_loc", "/tmp/artifacts")
            )
        ),
        runtime=data.get("runtime", {}),
        source=source,
    )
