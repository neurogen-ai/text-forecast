"""Regression tests pinning the INTENDED length-trim behaviour of
``data.preprocess.clean.main`` (plan 2.1 T0).

Intent:
  - level >= 3 (std filter): rows SHORTER than ``low`` are nulled.
  - level >= 4: rows LONGER than ``high`` are additionally nulled.
where low = max(mean - 3*std, min_len) and high = mean + 3*std,
computed from the length distribution of the fixture itself.

The current implementation has both comparisons inverted, so these tests
are expected to FAIL until the signs are fixed.
"""

import polars as pl

from data.preprocess.clean import main

# Fixture: text content deliberately avoids every substring in
# Filter.exclude_quality / Filter.exclude_lang so only the length trim
# affects the rows.


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


def test_level_3_nulls_short_rows_only():
    lf = _fixture()
    out = main(lf=lf, col="text", min_len=10, level=3).collect()
    mean, std = _length_stats()
    min_len = 10
    low = max(mean - 3 * std, min_len)
    high = mean + 3 * std

    # Sanity: the fixture must actually exercise both sides of the band.
    assert low == 10.0
    assert any(n < low for n in set(LENGTHS))
    assert any(low <= n <= high for n in set(LENGTHS))

    lens = out.with_columns(pl.col("text").str.len_chars().alias("_len"))
    surviving = lens.filter(pl.col("text").is_not_null())["_len"].to_list()
    # Intended: exactly the in-band rows survive; short rows (< low) are null.
    expected = [n for n in LENGTHS if low <= n <= high]
    assert surviving == expected


def test_level_4_additionally_nulls_long_rows():
    lf = _fixture()
    out = main(lf=lf, col="text", min_len=10, level=4).collect()
    mean, std = _length_stats()
    low = max(mean - 3 * std, 10)
    high = mean + 3 * std

    assert any(n > high for n in set(LENGTHS))

    lens = out.with_columns(pl.col("text").str.len_chars().alias("_len"))
    surviving = lens.filter(pl.col("text").is_not_null())["_len"].to_list()
    # Intended: only in-band rows survive; short AND long rows are null.
    expected = [n for n in LENGTHS if low <= n <= high]
    assert surviving == expected
