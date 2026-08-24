"""Explicit truncation policy for list-column datasets (plan 1.4.4).

``truncate_method`` used to be a free-form ``str``, and ``"drop"`` silently
filtered out rows longer than ``max_len`` without a trace. This module makes
the policy an explicit enum and logs affected row counts either way.
"""

from __future__ import annotations

from functools import reduce
from logging import getLogger
from typing import Literal, Mapping

import polars as pl

from utils.logging import setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)

TruncateMethod = Literal["truncate", "drop"]


def apply_truncation_policy(
    lf: pl.LazyFrame,
    *,
    name: str,
    limits: Mapping[str, int],
    truncate: bool = True,
    method: TruncateMethod,
) -> pl.LazyFrame:
    """Apply the truncation policy to ``lf`` and log affected row counts.

    ``limits`` maps list-column names to their maximum length; a row is
    affected when any listed column exceeds its limit. With
    ``method="drop"`` those rows are removed from the frame; with
    ``method="truncate"`` they are kept (per-item truncation happens in the
    dataset's ``__getitem__``). Either way the count is logged.
    """
    if not truncate:
        return lf

    over = reduce(lambda acc, expr: acc | expr, (pl.col(col).list.len() > limit for col, limit in limits.items()))
    n_over: int | None = lf.select(over.sum()).collect(engine="streaming").item()

    if method == "drop":
        logger.info(f"{name}: dropping {n_over} rows exceeding {dict(limits)}")
        return lf.filter(~over)

    logger.info(f"{name}: {n_over} rows exceed {dict(limits)} and will be truncated per item")
    return lf
