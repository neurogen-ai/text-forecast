"""CLI op-spec parser for the preprocess app (plan 2.1, T4a).

Pure parsing: turns ordered ``--op`` values plus per-op param
accumulators (as Typer delivers repeated options, in call order) into a
tuple of :class:`~src.data.preprocess.steps.PipelineStep` specs.

Op spec format is ``<kind>:<colspec>`` where ``colspec`` is a comma-
separated list of column names, e.g.::

    --op "clean:text" --op "dropna:title,abstract" --op "embed:text"

Per-op params are consumed in order via cursors: each op of a kind takes
the next value from its kind's param lists; defaults apply once the list
is exhausted. ``tokeniser`` applies to the most recent tokenise op;
``embedder_key`` / ``embedder_kwargs_json`` to the most recent embed op.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, cast

import typer

from data.preprocess.steps import (
    CleanStep,
    DropNaStep,
    EmbedStep,
    LengthTrim,
    PipelineStep,
    TokeniseStep,
)

__all__ = ["_parse_ops"]

_KINDS = {"clean", "dropna", "tokenise", "embed"}
_LANG_POLICIES = {"mark", "drop", "off"}


def _bad(msg: str) -> typer.BadParameter:
    return typer.BadParameter(msg)


def _parse_ops(
    ops: list[str],
    *,
    lowercase: list[bool] | None = None,
    trim_min_chars: list[int] | None = None,
    trim_max_sigma: list[float] | None = None,
    require_terminal_period: list[bool] | None = None,
    lang_policy: list[str] | None = None,
    drop_quality: list[bool] | None = None,
    tokeniser: str | None = None,
    embedder_key: str | None = None,
    embedder_kwargs_json: str | None = None,
) -> tuple[PipelineStep, ...]:
    """Parse ordered op specs and param accumulators into pipeline steps.

    Raises :class:`typer.BadParameter` on any validation error.
    """
    lowercase = lowercase or []
    trim_min_chars = trim_min_chars or []
    trim_max_sigma = trim_max_sigma or []
    require_terminal_period = require_terminal_period or []
    lang_policy = lang_policy or []
    drop_quality = drop_quality or []

    for value in lang_policy:
        if value not in _LANG_POLICIES:
            raise _bad(
                f"invalid --lang-policy {value!r}; "
                f"expected one of {sorted(_LANG_POLICIES)}"
            )

    steps: list[PipelineStep] = []
    clean_count = 0
    last_tokenise: TokeniseStep | None = None
    last_embed: EmbedStep | None = None

    def _take(values: list[Any], default: Any, flag: str, i: int) -> Any:
        if i < len(values):
            return values[i]
        return default

    for spec in ops:
        if ":" not in spec:
            raise _bad(f"malformed op spec {spec!r}: expected '<kind>:<colspec>'")
        kind, _, colspec = spec.partition(":")
        if kind not in _KINDS:
            raise _bad(
                f"unknown op kind {kind!r} in {spec!r}; expected one of {sorted(_KINDS)}"
            )
        if not colspec or not colspec.replace(",", "").strip():
            raise _bad(f"malformed op spec {spec!r}: empty column specification")
        cols = tuple(c.strip() for c in colspec.split(","))
        if any(not c for c in cols):
            raise _bad(f"malformed op spec {spec!r}: empty column name")
        dupes = {c for c in cols if cols.count(c) > 1}
        if dupes:
            raise _bad(f"duplicate column(s) {sorted(dupes)} in op {spec!r}")

        if kind == "clean":
            i = clean_count
            clean_count += 1
            min_chars = _take(trim_min_chars, None, "--trim-min-chars", i)
            max_sigma = _take(trim_max_sigma, None, "--trim-max-sigma", i)
            if min_chars is not None or max_sigma is not None:
                trim = LengthTrim(min_chars=min_chars, max_sigma=max_sigma)
            else:
                trim = None
            steps.append(
                CleanStep(
                    col=cols[0],
                    lowercase=_take(lowercase, False, "--lowercase", i),
                    drop_quality=_take(drop_quality, True, "--drop-quality", i),
                    lang_policy=_take(lang_policy, "mark", "--lang-policy", i),
                    trim=trim,
                    require_terminal_period=_take(
                        require_terminal_period, False, "--require-terminal-period", i
                    ),
                )
            )
        elif kind == "tokenise":
            last_tokenise = TokeniseStep(
                cols=cols,
                tokeniser="whitespace",
            )
            steps.append(last_tokenise)
        elif kind == "embed":
            last_embed = EmbedStep(
                cols=cols,
                embedder_key="minilm",
                embedder_kwargs={},
            )
            steps.append(last_embed)
        else:  # dropna
            steps.append(DropNaStep(cols=cols))

    # Post-walk pairing validations and single-value param attachment.
    extras = [
        (name, len(values))
        for name, values in (
            ("--lowercase", lowercase),
            ("--trim-min-chars", trim_min_chars),
            ("--trim-max-sigma", trim_max_sigma),
            ("--require-terminal-period", require_terminal_period),
            ("--lang-policy", lang_policy),
            ("--drop-quality", drop_quality),
        )
        if len(values) > clean_count
    ]
    if extras:
        name, n = extras[0]
        raise _bad(f"{name} given {n} times but only {clean_count} clean op(s)")

    if clean_count == 0:
        supplied_clean_params = [
            (name, values)
            for name, values in (
                ("--lowercase", lowercase),
                ("--trim-min-chars", trim_min_chars),
                ("--trim-max-sigma", trim_max_sigma),
                ("--require-terminal-period", require_terminal_period),
                ("--lang-policy", lang_policy),
                ("--drop-quality", drop_quality),
            )
            if values
        ]
        if supplied_clean_params:
            names = ", ".join(n for n, _ in supplied_clean_params)
            raise _bad(f"{names} supplied before any clean op")

    if tokeniser is not None and last_tokenise is None:
        raise _bad("--tokeniser supplied but no tokenise op given")
    if (embedder_key is not None or embedder_kwargs_json is not None) and (
        last_embed is None
    ):
        raise _bad("--embedder-key/--embedder-kwargs-json supplied but no embed op given")

    # Single-value options apply to the most recent matching op.
    if tokeniser is not None and last_tokenise is not None:
        idx = steps.index(last_tokenise)
        steps[idx] = dataclasses.replace(last_tokenise, tokeniser=tokeniser)
    if embedder_key is not None or embedder_kwargs_json is not None:
        kwargs_out: dict[str, Any] = {}
        if embedder_kwargs_json:
            try:
                parsed = json.loads(embedder_kwargs_json)
            except json.JSONDecodeError as exc:
                raise _bad(f"invalid JSON in embedder kwargs: {exc}") from None
            if not isinstance(parsed, dict):
                raise _bad("embedder kwargs must be a JSON object")
            kwargs_out = cast(dict[str, Any], parsed)
        if last_embed is not None:
            idx = steps.index(last_embed)
            steps[idx] = dataclasses.replace(
            last_embed,
            embedder_key=embedder_key or last_embed.embedder_key,
            embedder_kwargs=kwargs_out or last_embed.embedder_kwargs,
        )

    return tuple(steps)
