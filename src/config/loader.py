"""Experiment module loading and name resolution."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from typing import Any

from utils.get_root_dir import get_root_dir

from .env import Env
from .experiment import Experiment
from .runtime import RunContext


_EXPERIMENTS_DIR = Path(__file__).with_name("experiments")
_CONFIG_TOML = get_root_dir(markers=("pyproject.toml",)) / "config" / "config.toml"


def available_experiments() -> list[str]:
    """Return the stem names of all experiment modules."""
    return [
        p.stem
        for p in _EXPERIMENTS_DIR.glob("*.py")
        if p.name != "__init__.py"
    ]


def _toml_experiment() -> str | None:
    try:
        with open(_CONFIG_TOML, "rb") as f:
            data = tomllib.load(f)
        return data.get("experiment")
    except Exception:
        return None


def resolve_experiment_name(cli: str | None = None) -> str:
    """Resolve experiment name: CLI > config.toml > hard crash with list."""
    if cli:
        return cli
    toml = _toml_experiment()
    if toml:
        return toml
    raise ValueError(
        "No experiment selected. Use --experiment/-e or set 'experiment' in "
        f"config/config.toml. Available: {available_experiments()}"
    )


def read_experiment_name(name: str) -> str:
    """Return the module-level ``experiment_name`` string without building.

    Parses the module AST so runtime wrappers can create the MLflow run
    client-side (plan 2.0 P6) before the experiment object graph exists. Only
    a plain string assignment at module level counts; anything else raises.
    """
    import ast

    module_path = _EXPERIMENTS_DIR / f"{name}.py"
    if not module_path.exists():
        raise ValueError(
            f"Unknown experiment {name!r}. Available: {available_experiments()}"
        )
    tree = ast.parse(module_path.read_text())
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target, node.value
        if (
            target is not None
            and target.id == "experiment_name"
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            return value.value
    raise ValueError(
        f"Experiment {name!r} must declare a module-level string "
        "'experiment_name'"
    )


def _load_module_from_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_experiment(
    name: str,
    runtime: RunContext,
    *,
    env: Env,
) -> Experiment[Any]:
    """Import an experiment module and call its ``build`` entry point."""
    module_path = _EXPERIMENTS_DIR / f"{name}.py"
    if not module_path.exists():
        raise ValueError(
            f"Unknown experiment {name!r}. Available: {available_experiments()}"
        )

    module = _load_module_from_path(f"config.experiments.{name}", module_path)

    experiment_name = getattr(module, "experiment_name", None)
    build = getattr(module, "build", None)

    if not isinstance(experiment_name, str):
        raise ValueError(
            f"Experiment {name!r} must declare a string 'experiment_name'"
        )
    if not callable(build):
        raise ValueError(
            f"Experiment {name!r} must define a callable 'build(runtime, env)'"
        )

    experiment = build(runtime, env)
    if not isinstance(experiment, Experiment):
        raise ValueError(
            f"Experiment {name!r}.build() must return an Experiment instance"
        )
    return experiment


def load_experiment_from_path(
    path: Path,
    runtime: RunContext,
    *,
    env: Env,
) -> Experiment[Any]:
    """Load an experiment module from an arbitrary path (used by eval's
    downloaded experiment-file artifact).
    """
    if not path.exists():
        raise FileNotFoundError(f"Experiment file not found: {path}")

    module_name = f"config.experiments._artifact.{path.stem}"
    module = _load_module_from_path(module_name, path)

    experiment_name = getattr(module, "experiment_name", None)
    build = getattr(module, "build", None)

    if not isinstance(experiment_name, str):
        raise ValueError("Experiment file must declare a string 'experiment_name'")
    if not callable(build):
        raise ValueError(
            "Experiment file must define a callable 'build(runtime, env)'"
        )

    experiment = build(runtime, env)
    if not isinstance(experiment, Experiment):
        raise ValueError("Experiment file build() must return an Experiment instance")
    return experiment
