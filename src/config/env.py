"""Environment / machine settings.

Typed :class:`Env` dataclass loaded from ``config/config.toml`` with optional
CLI overrides.  No module-level singletons or PEP 562 shims remain.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.get_root_dir import get_root_dir


@dataclass(frozen=True, kw_only=True)
class Env:
    """Machine-specific settings that are not experiment-relevant."""

    tracking_uri: str
    raw_loc: Path
    staged_loc: Path
    artifact_loc: Path


def _load_toml() -> dict[str, Any]:
    path = get_root_dir(markers=("pyproject.toml",)) / "config" / "config.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_env(overrides: dict[str, Any] | None = None) -> Env:
    """Load ``[env]`` from ``config/config.toml`` and apply CLI overrides.

    All four keys get CLI override flags.  ``staged_loc`` is validated to
    exist at load time.
    """
    overrides = overrides or {}
    data = _load_toml()
    env_data = data.get("env", {})

    staged_loc = Path(
        overrides.get("staged_loc", env_data.get("staged_loc", "/tmp/staged"))
    )
    if not staged_loc.exists():
        raise FileNotFoundError(f"staged_loc does not exist: {staged_loc}")

    return Env(
        tracking_uri=overrides.get(
            "tracking_uri", env_data.get("tracking_uri", "http://127.0.0.1:5000")
        ),
        raw_loc=Path(
            overrides.get("raw_loc", env_data.get("raw_loc", "/tmp/raw"))
        ),
        staged_loc=staged_loc,
        artifact_loc=Path(
            overrides.get(
                "artifact_loc", env_data.get("artifact_loc", "/tmp/artifacts")
            )
        ),
    )
