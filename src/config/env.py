"""Environment / machine settings.

Typed :class:`Env` dataclass loaded from ``config/config.toml`` with optional
CLI overrides.  No module-level singletons or PEP 562 shims remain.
"""

from __future__ import annotations

import tomllib
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.get_root_dir import get_root_dir


@dataclass(frozen=True, kw_only=True)
class ModalRuntimeConfig:
    """Typed view of ``[runtime.modal]`` from ``config/config.toml``.

    Defaults mirror the historical module-level fallbacks in
    ``runtime.modal_runtime``. ``tracking_uri`` is ``None`` when unset, meaning
    inherit ``[env].tracking_uri``.
    """

    project: str = "citef"
    gpu: str = "L4"
    train_gpu: str = "A10G"
    embedder_batch_size: int = 64
    python_version: str = "3.12"
    raw_volume: str = "openalex-raw"
    staged_volume: str = "openalex-staged"
    checkpoint_volume: str | None = None
    timeout: int = 86400
    tracking_uri: str | None = None


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


def modal_runtime_config(
    env: Env, *, require_checkpoint_volume: bool = False
) -> ModalRuntimeConfig:
    """Build a typed :class:`ModalRuntimeConfig` from ``env.runtime["modal"]``.

    Set ``require_checkpoint_volume`` for train/eval dispatch paths: without a
    checkpoint volume there is nowhere to stage checkpoint scratch inside the
    container, so the call fails fast with the key to set.
    """

    raw = env.runtime.get("modal", {})
    if not isinstance(raw, dict):
        raise ValueError("[runtime.modal] must be a table in config.toml")

    known: dict[str, Any] = {}
    for f in dataclasses.fields(ModalRuntimeConfig):
        if f.name in raw:
            known[f.name] = raw[f.name]
    unknown = sorted(set(raw) - {f.name for f in dataclasses.fields(ModalRuntimeConfig)})
    if unknown:
        raise ValueError(f"Unknown keys in [runtime.modal]: {unknown}")

    cfg = ModalRuntimeConfig(**known)
    if require_checkpoint_volume and not cfg.checkpoint_volume:
        raise ValueError(
            "[runtime.modal].checkpoint_volume is required for remote "
            "train/eval. Add it to config.toml under [runtime.modal]."
        )
    return cfg
