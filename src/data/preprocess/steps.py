"""Step specs for the preprocess pipeline (plan 2.1 §2.1).

Pure specification module: frozen dataclasses describing pipeline
operations, plus serialisation helpers. No consumer logic lives here.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Union, cast, get_type_hints

__all__ = [
    "CleanStep",
    "DropNaStep",
    "EmbedStep",
    "LengthTrim",
    "PipelineStep",
    "StepContext",
    "TokeniseStep",
    "step_from_dict",
    "step_to_dict",
]


@dataclass(frozen=True)
class LengthTrim:
    """Length-based row trimming spec."""

    min_chars: int | None = None  # hard minimum, replaces min_len
    max_sigma: float | None = None  # mean + sigma * max_sigma upper cut


@dataclass(frozen=True, kw_only=True)
class CleanStep:
    tag: Literal["clean"] = "clean"
    col: str
    lowercase: bool = False
    drop_quality: bool = True  # Filter.exclude_quality substrings -> null
    lang_policy: Literal["mark", "drop", "off"] = "mark"
    trim: LengthTrim | None = None
    require_terminal_period: bool = False


@dataclass(frozen=True, kw_only=True)
class TokeniseStep:
    tag: Literal["tokenise"] = "tokenise"
    tokeniser: str
    cols: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class EmbedStep:
    tag: Literal["embed"] = "embed"
    embedder_key: str
    embedder_kwargs: dict[str, Any] = field(default_factory=dict[str, Any])
    cols: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class DropNaStep:
    tag: Literal["drop_na"] = "drop_na"
    cols: tuple[str, ...]


PipelineStep = Union[CleanStep, TokeniseStep, EmbedStep, DropNaStep]

_STEP_TYPES: dict[str, type[PipelineStep]] = {
    s.tag: s  # type: ignore[attr-defined]
    for s in PipelineStep.__args__  # type: ignore[attr-defined]
}


@dataclass
class StepContext:
    """Runtime context handed to step handlers.

    Non-frozen by design: it holds mutable runtime objects.
    """

    runtime: Any  # Runtime protocol from src/runtime/base.py when typed
    embedder: Any
    partition_index: int
    logger: logging.Logger


def step_to_dict(step: PipelineStep) -> dict[str, Any]:
    """Serialise a step to a JSON-compatible dict (tag included)."""
    return asdict(step)


def _coerce(name: str, cls: type[PipelineStep], value: Any) -> Any:
    """Reconstruct nested dataclass fields (e.g. CleanStep.trim) from dicts."""
    typ = get_type_hints(cls)[name]
    inner = getattr(typ, "__args__", ())
    types = tuple(t for t in (typ, *inner) if dataclasses.is_dataclass(t))
    if types and isinstance(value, dict):
        return cast(Any, types[0])(**value)
    return value


def step_from_dict(d: dict[str, Any]) -> PipelineStep:
    """Reconstruct a step from its dict form, dispatching on ``tag``."""
    try:
        tag = d["tag"]
    except KeyError:
        raise ValueError(f"step dict missing 'tag' key: {d!r}") from None
    try:
        cls = _STEP_TYPES[tag]
    except KeyError:
        raise ValueError(
            f"unknown step tag {tag!r}; expected one of {sorted(_STEP_TYPES)}"
        ) from None
    kwargs = {k: _coerce(k, cls, v) for k, v in d.items() if k != "tag"}
    names = {f.name for f in fields(cls)}
    unknown = set(kwargs) - names
    if unknown:
        raise ValueError(f"unknown fields {sorted(unknown)} for step {tag!r}")
    if cls is EmbedStep and "embedder_kwargs" not in kwargs:
        kwargs["embedder_kwargs"] = {}
    return cls(**kwargs)  # type: ignore[return-value]
