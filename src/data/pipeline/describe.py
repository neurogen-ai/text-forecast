"""Runtime-agnostic describe pipeline.

Extracted from ``apps.describe`` so the logic can run locally or inside a
Modal container without the app knowing which.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from logging import getLogger

import polars as pl

from data.sources.base import DataSource
from utils.logging import setup_logger


logger = getLogger(__name__)
_ = setup_logger(logger)


@dataclass(frozen=True, kw_only=True)
class DescribeJob:
    """Fully serialisable description of a describe job.

    The job contains no runtime state; it is passed to
    ``Runtime.run_describe`` (step 3 of plan 1.3) which decides whether to
    execute locally or dispatch to a remote container.
    """

    source: DataSource
    start_time: datetime | None = None
    end_time: datetime | None = None
    time_col: str = "publication_date"
    describe_cols: list[str] = field(default_factory=list)
    buckets: list[float] = field(default_factory=list)
    filter_expr: str | None = None


def run_describe_pipeline(job: DescribeJob) -> None:
    """Log bucket counts/proportions/weights and column descriptives."""
    source_path = job.source.resolve()
    logger.info(f"Describing dataset at {source_path}")

    lf = pl.scan_parquet(list(source_path.glob("*.par*")))

    if job.start_time is not None:
        lf = lf.filter(pl.col(job.time_col) >= job.start_time)
    if job.end_time is not None:
        lf = lf.filter(pl.col(job.time_col) < job.end_time)

    if job.filter_expr:
        lf = lf.filter(eval(job.filter_expr))

    buckets = job.buckets
    n_buckets = len(buckets)
    for col in job.describe_cols:
        bucket_counts: dict[str, int | float] = {
            f"{buckets[i]}-{buckets[i + 1]}": None for i in range(n_buckets - 1)
        }
        weights = {k: v for k, v in bucket_counts.items()}
        for i in range(n_buckets - 1):
            bucket_counts[f"{buckets[i]}-{buckets[i + 1]}"] = (
                lf.filter(
                    (pl.col(col) >= buckets[i]) & (pl.col(col) < buckets[i + 1])
                )
                .select(pl.len())
                .collect(engine="streaming")
                .item()
            )
        total = sum([v for v in bucket_counts.values()])

        for k in weights.keys():
            if bucket_counts[k] == 0:
                weights[k] = float("nan")
                continue
            weights[k] = total / ((n_buckets - 1) * bucket_counts[k])

        proportions = {k: round((v / total) * 100, 1) for k, v in bucket_counts.items()}

        print()
        logger.info(f"Describing {col}")
        logger.info("counts: ")

        logger.info([f"{k}:" + f"{v:,}" for k, v in bucket_counts.items()])

        logger.info("proportions: ")
        logger.info([f"{k}:" + f"{v:,}" for k, v in proportions.items()])

        logger.info("weights: ")
        logger.info(weights)

        description = lf.select(pl.col(col)).describe()
        logger.info(description)

    logger.info("Finished Describe")
