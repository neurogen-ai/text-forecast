"""Scalar param collection and guarded MLflow param logging.

The walker owns "what counts as a param": a new param source is one root at
the call site, a new field on any config dataclass or pydantic model is
picked up with no code change, and a new exotic field type is one branch in
the scalar filter. Nothing outside this module knows how collection works.

Contracts live in plans/implementation/v2.3.1.md (Foundational contracts);
the guarded log exists because the train pipeline reuses job.run_id on
resume (AD9), and MLflow params are immutable per run.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import fields as dataclass_fields, is_dataclass
from typing import NamedTuple

import torch

from .param_keys import SUFFIX_RULES

logger = logging.getLogger(__name__)

LeafValue = bool | int | float | str

# MLflow caps param key and value lengths (mlflow 3.x validation limits:
# 250 chars for keys, 6000 for values). Truncate instead of raising; only
# exotic fields can hit this (release doc, friction points).
MLFLOW_MAX_PARAM_KEY_LENGTH = 250
MLFLOW_MAX_PARAM_VALUE_LENGTH = 6000

_SCALAR_TYPES = (bool, int, float, str)


class Leaf(NamedTuple):
    value: LeafValue
    origin: str  # component path, e.g. "model.config.embedder" or "train"


try:  # pydantic support degrades gracefully in dataclass-only environments
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - exercised only without pydantic
    BaseModel = None  # type: ignore[assignment, misc]


def _is_pydantic_model(obj: object) -> bool:
    return BaseModel is not None and isinstance(obj, BaseModel)


def _instance_fields(obj: object) -> list[str]:
    if is_dataclass(obj) and not isinstance(obj, type):
        return [f.name for f in dataclass_fields(obj)]  # type: ignore[arg-type]
    if _is_pydantic_model(obj):
        return list(type(obj).model_fields)
    return []


def _add_leaf(
    leaves: dict[str, Leaf], value: LeafValue, path: tuple[str, ...]
) -> None:
    origin = ".".join(path)
    bare_key = path[-1]
    existing = leaves.get(origin)
    if existing is not None:
        if existing.value == value:
            return
        raise KeyError(
            f"colliding paths {existing.origin!r} and {origin!r} share the "
            f"bare key {bare_key!r} but carry different values "
            f"({existing.value!r} vs {value!r})"
        )
    for other_origin, other in leaves.items():
        other_bare = other_origin.rsplit(".", 1)[-1]
        if other_bare == bare_key and other.value != value:
            if bare_key in SUFFIX_RULES:
                break  # tabled ambiguity; resolve_keys disambiguates
            raise KeyError(
                f"colliding paths {other_origin!r} and {origin!r} share the "
                f"bare key {bare_key!r} but carry different values "
                f"({other.value!r} vs {value!r})"
            )
    leaves[origin] = Leaf(value=value, origin=origin)


def _visit(obj: object, path: tuple[str, ...], leaves: dict[str, Leaf]) -> None:
    if obj is None:
        _add_leaf(leaves, "None", path)
    elif isinstance(obj, _SCALAR_TYPES):
        _add_leaf(leaves, obj, path)
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            _visit(value, (*path, str(key)), leaves)
    elif (fields := _instance_fields(obj)):
        for name in fields:
            _visit(getattr(obj, name, None), (*path, name), leaves)
    elif isinstance(obj, (torch.device, torch.dtype)):
        # Runtime scalars whose str() form IS the information ("cuda:0",
        # "torch.bfloat16"). torch.dtype.__name__ is just "dtype", so the
        # class-name fallback would record nothing about the run.
        _add_leaf(leaves, str(obj), path)
    elif isinstance(obj, (tuple, list)) and all(
        isinstance(x, _SCALAR_TYPES) for x in obj
    ):
        # Containers of scalars (e.g. WarmupCosineSpec.milestones) record
        # their repr instead of the useless "tuple"/"list" class name.
        _add_leaf(leaves, repr(obj), path)
    else:
        # Live objects (model, tracker, stream, ...) stringify as their
        # class name so run start never raises on a config that holds them
        # (release doc, friction points).
        _add_leaf(leaves, type(obj).__name__, path)


def collect_scalars(*roots: tuple[str, object]) -> dict[str, Leaf]:
    """Collect scalar leaves from param roots, keyed by component path.

    Recurse dataclass and pydantic BaseModel instances field by field;
    treat Mapping[str, scalar] roots as explicit leaves. Keep
    str/int/float/bool leaves; stringify torch devices/dtypes via str()
    and containers of scalars via repr(); stringify other objects as their
    class name, with None as "None". Raise KeyError naming the colliding
    paths when two bare leaves carry different values, except for bare
    keys named in SUFFIX_RULES, which resolve_keys disambiguates.

    The returned dict is keyed by full origin path, so a tabled leaf may
    appear under several origins with differing values.
    """
    leaves: dict[str, Leaf] = {}
    for name, obj in roots:
        _visit(obj, (name,), leaves)
    return leaves


def log_params_guarded(params: dict[str, str]) -> None:
    """Log params to the active MLflow run, truncating values to the MLflow
    cap instead of raising. Identical values are a no-op; changed values log
    a warning and keep the original (params are immutable per run). Empty
    values are skipped with a warning: some MLflow backends reject them
    ("Param value cannot be empty"), and a rejected log_params would raise
    out of run start, the one failure mode this guard exists to prevent.
    """
    import mlflow  # lazy so importing this module needs no mlflow

    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("log_params_guarded requires an active MLflow run")
    existing: dict[str, str] = mlflow.get_run(run.info.run_id).data.params
    payload: dict[str, str] = {}
    for raw_key, raw_value in params.items():
        key = str(raw_key)[:MLFLOW_MAX_PARAM_KEY_LENGTH]
        value = str(raw_value)[:MLFLOW_MAX_PARAM_VALUE_LENGTH]
        if value == "":
            logger.warning(
                "param %s has an empty value; skipping it (some MLflow "
                "backends reject empty param values)",
                key,
            )
            continue
        if key in existing:
            if existing[key] == value:
                continue
            logger.warning(
                "param %s already logged as %r on this run; keeping the "
                "original and ignoring %r (params are immutable per run; "
                "a changed value on resume means the run_id should have "
                "been new)",
                key,
                existing[key],
                value,
            )
            continue
        payload[key] = value
    if payload:
        mlflow.log_params(payload)
