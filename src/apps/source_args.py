"""Shared helpers for selecting and building a source backend from CLI flags."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from config.env import Env
from data.sources import SourceBackend, build_source_backend


def source_backend_arg(default: str | None = None):
    return typer.Option(
        default,
        "--source-backend",
        help="Data source backend (defaults to [source].default)",
    )


def source_opt_arg():
    return typer.Option(
        [],
        "--source-opt",
        help="Backend-specific option as key=value (can be repeated)",
    )


def source_base_dir_arg():
    return typer.Option(
        None,
        "--source-base-dir",
        help="Shortcut for --source-backend local --source-opt base_dir=...",
    )


def source_volume_arg():
    return typer.Option(
        None,
        "--source-volume",
        help="Shortcut for --source-backend modal --source-opt volume=...",
    )


def _parse_source_opts(opts: list[str]) -> dict[str, Any]:
    """Parse ``--source-opt key=value`` flags into a nested dict."""
    result: dict[str, Any] = {}
    for opt in opts:
        if "=" not in opt:
            raise typer.BadParameter(
                f"Invalid --source-opt {opt!r}; expected key=value"
            )
        key, value = opt.split("=", 1)
        if value.isdigit():
            value = int(value)
        parts = key.split(".")
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def build_source_backend_from_cli(
    env: Env,
    source_backend: str | None,
    source_opts: list[str],
    source_base_dir: Path | None,
    source_volume: str | None,
) -> SourceBackend:
    """Resolve CLI source flags into a configured :class:`SourceBackend`."""
    backend = source_backend or env.source.get("default", "local")
    overrides = _parse_source_opts(source_opts)

    if source_base_dir is not None:
        if source_backend is not None and source_backend != "local":
            raise typer.BadParameter(
                "--source-base-dir implies --source-backend local"
            )
        backend = "local"
        overrides.setdefault("local", {})["base_dir"] = str(source_base_dir)

    if source_volume is not None:
        if source_backend is not None and source_backend != "modal":
            raise typer.BadParameter(
                "--source-volume implies --source-backend modal"
            )
        backend = "modal"
        overrides.setdefault("modal", {})["volume"] = source_volume

    config = env.source.get(backend, {}).copy()
    _deep_update(config, overrides.get(backend, {}))

    return build_source_backend(backend, config)
