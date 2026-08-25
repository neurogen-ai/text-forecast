"""Tests for the preprocess CLI op-spec parser (plan 2.1, T4a)."""

from __future__ import annotations

import json

import pytest
import typer

from src.apps.preprocess_ops import _parse_ops


def test_ordering_across_multiple_ops() -> None:
    steps = _parse_ops(
        ["clean:text", "tokenise:title,abstract", "dropna:title,abstract", "embed:text"]
    )
    assert [type(s).__name__ for s in steps] == [
        "CleanStep",
        "TokeniseStep",
        "DropNaStep",
        "EmbedStep",
    ]
    assert steps[1].cols == ("title", "abstract")
    assert steps[2].cols == ("title", "abstract")


def test_params_attach_to_correct_clean_op_in_order() -> None:
    steps = _parse_ops(
        ["clean:a", "clean:b"],
        lowercase=[True, False],
        trim_min_chars=[20, None],
        trim_max_sigma=[None, 3.0],
        lang_policy=["off", "drop"],
        drop_quality=[False, True],
        require_terminal_period=[True, False],
    )
    first, second = steps
    assert first.col == "a"
    assert first.lowercase is True
    assert first.lang_policy == "off"
    assert first.drop_quality is False
    assert first.require_terminal_period is True
    assert first.trim is not None and first.trim.min_chars == 20
    assert second.col == "b"
    assert second.lowercase is False
    assert second.lang_policy == "drop"
    assert second.drop_quality is True
    assert second.require_terminal_period is False
    assert second.trim is not None and second.trim.max_sigma == 3.0


def test_defaults_when_params_exhausted_or_absent() -> None:
    steps = _parse_ops(["clean:a", "clean:b"], lowercase=[True])
    s1, s2 = steps
    assert s1.lowercase is True
    assert s2.lowercase is False
    assert s1.trim is None
    assert s2.trim is None
    assert s1.lang_policy == "mark"
    assert s1.drop_quality is True
    # tokenise/embed defaults when no tokeniser/embedder given.
    tsteps = _parse_ops(["tokenise:t", "embed:e"])
    assert tsteps[0].tokeniser == "whitespace"
    assert tsteps[1].embedder_key == "minilm"
    assert tsteps[1].embedder_kwargs == {}


def test_tokeniser_and_embedder_attach_to_last_matching_op() -> None:
    steps = _parse_ops(
        ["tokenise:a", "tokenise:b", "embed:x,y", "embed:z"],
        tokeniser="spacy",
        embedder_key="bge",
        embedder_kwargs_json=json.dumps({"batch_size": 16}),
    )
    assert steps[0].tokeniser == "whitespace"
    assert steps[1].tokeniser == "spacy"
    assert steps[2].embedder_key == "minilm"
    assert steps[2].embedder_kwargs == {}
    assert steps[3].embedder_key == "bge"
    assert steps[3].embedder_kwargs == {"batch_size": 16}


# ---- error paths ----


def test_unknown_op_kind() -> None:
    with pytest.raises(typer.BadParameter, match="unknown op kind"):
        _parse_ops(["frobnicate:text"])


@pytest.mark.parametrize("spec", ["no-colon-here", "clean:", "clean:   "])
def test_malformed_op_spec(spec: str) -> None:
    with pytest.raises(typer.BadParameter, match="malformed op spec"):
        _parse_ops([spec])


def test_duplicate_columns_within_op() -> None:
    with pytest.raises(typer.BadParameter, match="duplicate column"):
        _parse_ops(["dropna:title,title"])


@pytest.mark.parametrize("kwargs", [
    {"lowercase": [True]},
    {"trim_min_chars": [10]},
    {"trim_max_sigma": [3.0]},
    {"require_terminal_period": [True]},
    {"lang_policy": ["drop"]},
    {"drop_quality": [False]},
])
def test_clean_params_before_any_clean_op(kwargs: dict) -> None:
    with pytest.raises(typer.BadParameter):
        _parse_ops(["dropna:text"], **kwargs)


@pytest.mark.parametrize("field", ["lowercase", "trim_min_chars", "trim_max_sigma",
                                   "require_terminal_period", "lang_policy",
                                   "drop_quality"])
def test_more_clean_values_than_clean_ops(field: str) -> None:
    kwargs = {field: [True] if field in ("lowercase", "require_terminal_period",
                                         "drop_quality") else ([3] if field.startswith("trim_min") else ([3.0] if field.startswith("trim_max") else ["off"]))}
    with pytest.raises(typer.BadParameter, match="times but only"):
        _parse_ops([], **kwargs)


def test_tokeniser_without_tokenise_op() -> None:
    with pytest.raises(typer.BadParameter, match="no tokenise op"):
        _parse_ops(["clean:text"], tokeniser="spacy")


def test_embedder_without_embed_op() -> None:
    with pytest.raises(typer.BadParameter, match="no embed op"):
        _parse_ops(["clean:text"], embedder_key="bge")
    with pytest.raises(typer.BadParameter, match="no embed op"):
        _parse_ops(["clean:text"], embedder_kwargs_json="{}")


def test_invalid_lang_policy() -> None:
    with pytest.raises(typer.BadParameter, match="invalid --lang-policy"):
        _parse_ops(["clean:text"], lang_policy=["yolo"])


def test_invalid_embedder_kwargs_json() -> None:
    with pytest.raises(typer.BadParameter, match="invalid JSON"):
        _parse_ops(["embed:text"], embedder_kwargs_json="{not json")
    with pytest.raises(typer.BadParameter, match="JSON object"):
        _parse_ops(["embed:text"], embedder_kwargs_json="[1]")
