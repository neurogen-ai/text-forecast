"""Unit tests for step spec serialisation (T1a)."""

import dataclasses

import pytest

from src.data.preprocess.steps import (
    CleanStep,
    DropNaStep,
    EmbedStep,
    LengthTrim,
    TokeniseStep,
    step_from_dict,
    step_to_dict,
)


def test_clean_step_roundtrip() -> None:
    s = CleanStep(col="text")
    assert step_from_dict(step_to_dict(s)) == s


def test_clean_step_with_trim_roundtrip() -> None:
    s = CleanStep(
        col="title",
        lowercase=True,
        drop_quality=False,
        lang_policy="drop",
        trim=LengthTrim(min_chars=10, max_sigma=3.0),
        require_terminal_period=True,
    )
    assert step_from_dict(step_to_dict(s)) == s


def test_tokenise_step_roundtrip() -> None:
    s = TokeniseStep(tokeniser="whitespace", cols=("a", "b"))
    assert step_from_dict(step_to_dict(s)) == s


def test_embed_step_roundtrip() -> None:
    s = EmbedStep(
        embedder_key="huggingface",
        embedder_kwargs={"batch_size": 32, "device": "cpu"},
        cols=("text",),
    )
    d = step_to_dict(s)
    assert isinstance(d["embedder_kwargs"], dict)
    rt = step_from_dict(d)
    assert rt == s
    assert rt.embedder_kwargs == {"batch_size": 32, "device": "cpu"}


def test_embed_step_default_kwargs_roundtrip() -> None:
    s = EmbedStep(embedder_key="modal", cols=("x",))
    assert step_from_dict(step_to_dict(s)) == s


def test_drop_na_step_roundtrip() -> None:
    s = DropNaStep(cols=("a",))
    assert step_from_dict(step_to_dict(s)) == s


@pytest.mark.parametrize("cls", [CleanStep, TokeniseStep, EmbedStep, DropNaStep])
def test_steps_are_frozen(cls) -> None:  # type: ignore[no-untyped-def]
    assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_unknown_tag_raises() -> None:
    with pytest.raises(ValueError, match="unknown step tag"):
        step_from_dict({"tag": "nope"})


def test_missing_tag_raises() -> None:
    with pytest.raises(ValueError, match="missing 'tag'"):
        step_from_dict({"col": "text"})


def test_length_trim_defaults_none() -> None:
    t = LengthTrim()
    assert t.min_chars is None and t.max_sigma is None
    assert dataclasses.asdict(t) == {"min_chars": None, "max_sigma": None}
