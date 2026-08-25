"""Unit + regression tests for data.preprocess.clean (plan 2.1 T1b).

run_clean operates from explicit CleanStep flags; the two former regression
tests (T0) now target the handler with corrected trim semantics:
  - rows shorter than ``min_chars`` are dropped (hard minimum);
  - when ``max_sigma`` is set, rows longer than mean + sigma * std are
    dropped; no lower statistical bound.
"""

import polars as pl
import pytest

from data.preprocess.clean import EXCLUDE_LANG, EXCLUDE_QUALITY, run_clean
from data.preprocess.steps import CleanStep, LengthTrim

# Fixture text avoids every substring in EXCLUDE_QUALITY / EXCLUDE_LANG so
# only the requested flags affect the rows.


def _text(n: int) -> str:
    base = "lorem ipsum "
    return (base * (n // len(base) + 1))[:n]


LENGTHS = [5] * 10 + [30] * 28 + [1000] * 2  # 40 rows


def _fixture() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "text": [_text(n) for n in LENGTHS],
            "language": ["en"] * len(LENGTHS),
        }
    )


def _length_stats() -> tuple[float, float]:
    s = pl.Series(LENGTHS)
    return float(s.mean()), float(s.std())


def _surviving_lengths(out: pl.DataFrame) -> list[int]:
    return (
        out.filter(pl.col("text").is_not_null())
        .with_columns(pl.col("text").str.len_chars().alias("_len"))["_len"]
        .to_list()
    )


# ---------------------------------------------------------------- dedup checks


def test_constants_deduplicated():
    assert len(EXCLUDE_QUALITY) == len(set(EXCLUDE_QUALITY))
    assert len(EXCLUDE_LANG) == len(set(EXCLUDE_LANG))


# ------------------------------------------------------- trim regression (T0)


def test_trim_min_only_drops_short_rows():
    lf = _fixture()
    step = CleanStep(col="text", trim=LengthTrim(min_chars=10))
    out = run_clean(lf, step).collect()

    assert any(n < 10 for n in set(LENGTHS))
    expected = [n for n in LENGTHS if n >= 10]
    assert _surviving_lengths(out) == expected


def test_trim_sigma_only_drops_long_rows():
    lf = _fixture()
    mean, std = _length_stats()
    high = mean + 3 * std

    step = CleanStep(col="text", trim=LengthTrim(max_sigma=3.0))
    out = run_clean(lf, step).collect()

    assert any(n > high for n in set(LENGTHS))
    expected = [n for n in LENGTHS if n <= high]
    assert _surviving_lengths(out) == expected


def test_trim_min_and_sigma_combined():
    lf = _fixture()
    mean, std = _length_stats()
    high = mean + 3 * std

    step = CleanStep(col="text", trim=LengthTrim(min_chars=10, max_sigma=3.0))
    out = run_clean(lf, step).collect()

    expected = [n for n in LENGTHS if 10 <= n <= high]
    assert _surviving_lengths(out) == expected


def test_level_4_flags_match_corrected_semantics():
    """The old level-4 mapping now agrees with the intended band semantics."""
    lf = _fixture()
    mean, std = _length_stats()
    high = mean + 3 * std

    step = CleanStep(
        col="text",
        lowercase=True,
        trim=LengthTrim(min_chars=10, max_sigma=3.0),
    )
    out = run_clean(lf, step).collect()
    expected = [n for n in LENGTHS if 10 <= n <= high]
    assert _surviving_lengths(out) == expected


# ------------------------------------------------------------- per-flag tests


def test_lowercase_flag():
    lf = pl.LazyFrame({"text": ["Hello WORLD"], "language": ["en"]})
    out = run_clean(lf, CleanStep(col="text", lowercase=True)).collect()
    assert out["text"].to_list() == ["hello world"]
    assert out["language"].to_list() == ["en"]


def test_drop_quality_nulls_matching_rows():
    lf = pl.LazyFrame(
        {
            "text": ["fine text here", "see http link", "also fine"],
            "language": ["en"] * 3,
        }
    )
    out = run_clean(lf, CleanStep(col="text")).collect()
    assert out["text"].to_list() == ["fine text here", None, "also fine"]
    assert out["language"].to_list() == ["en", "en", "en"]


def test_drop_quality_off_keeps_all():
    lf = pl.LazyFrame(
        {"text": ["see http link"], "language": ["en"]}
    )
    out = run_clean(lf, CleanStep(col="text", drop_quality=False)).collect()
    assert out["text"].to_list() == ["see http link"]


@pytest.mark.parametrize("policy", ["mark", "drop", "off"])
def test_lang_policy(policy):
    # 的 is in EXCLUDE_LANG; plain English row is not.
    lf = pl.LazyFrame(
        {
            "text": ["clean english words", "中文的文本"],
            "language": ["en", "zh"],
        }
    )
    out = run_clean(lf, CleanStep(col="text", lang_policy=policy)).collect()
    if policy == "mark":
        assert out["language"].to_list() == ["en", "unknown"]
        assert out.height == 2
    elif policy == "drop":
        assert out["language"].to_list() == ["en"]
        assert out.height == 1
    else:
        assert out["language"].to_list() == ["en", "zh"]
        assert out.height == 2


def test_require_terminal_period():
    lf = pl.LazyFrame(
        {"text": ["ends with period.", "no period here", "other."], "language": ["en"] * 3}
    )
    out = run_clean(
        lf, CleanStep(col="text", require_terminal_period=True)
    ).collect()
    assert sorted(out["text"].to_list()) == ["ends with period.", "other."]


def test_combined_flags_and_lazy_output():
    lf = _fixture()
    step = CleanStep(
        col="text",
        lowercase=True,
        lang_policy="drop",
        trim=LengthTrim(min_chars=10),
    )
    result = run_clean(lf, step)
    assert isinstance(result, pl.LazyFrame)
    out = result.collect()
    lengths = {n for n in LENGTHS if n >= 10}
    assert set(_surviving_lengths(out)) <= lengths


def test_run_clean_accepts_ctx_argument():
    lf = pl.LazyFrame({"text": ["abc"], "language": ["en"]})
    out = run_clean(lf, CleanStep(col="text"), ctx=None).collect()
    assert out["text"].to_list() == ["abc"]
