"""Runtime-agnostic engineer pipeline.

Extracted from ``apps.engineer`` so the logic can run locally or inside a
Modal container without the app knowing which.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from logging import getLogger

import numpy as np
import polars as pl
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sklearn.cluster import DBSCAN

from data.sources.base import DataSource
from utils.logging import setup_logger


logger = getLogger(__name__)
_ = setup_logger(logger)


@dataclass(frozen=True, kw_only=True)
class EngineerJob:
    """Fully serialisable description of an engineer job.

    The job contains no runtime state; it is passed to
    ``Runtime.run_engineer`` (step 3 of plan 1.3) which decides whether to
    execute locally or dispatch to a remote container.
    """

    origin: DataSource
    output: DataSource
    len_cols: list[str] = field(default_factory=list)
    years_to_first: bool = True
    n_partitions: int = 64
    dry_run: bool = False


def run_dbscan_on_chunk(df_chunk: pl.DataFrame) -> pl.DataFrame:
    """Helper to process a single materialized chunk row-by-row."""
    results = []
    for row in df_chunk.iter_rows(named=True):
        vals = row["counts_per_year_delta"]
        freqs = row["counts_per_year_counts"]

        if not vals:
            continue

        X = np.array(vals).reshape(-1, 1)
        weights = np.array(freqs)

        db = DBSCAN(eps=15, min_samples=5).fit(X, sample_weight=weights)
        labels = db.labels_

        unique_labels = set(labels) - {-1}

        total_weight = weights.sum()
        noise_weight = weights[labels == -1].sum() if -1 in labels else 0
        noise_ratio = noise_weight / total_weight if total_weight > 0 else 0.0

        centroids = []
        n_members = []
        for label in sorted(unique_labels):
            mask = labels == label
            centroid = np.average(X[mask], axis=0, weights=weights[mask])
            centroids.append(float(centroid[0]))
            n_members.append(sum(labels == label))

        results.append(
            {
                "id": row["id"],
                "n_clusters": len(unique_labels),
                "centroids": centroids,
                "n_members": n_members,
                "noise_ratio": noise_ratio,
            }
        )
    return pl.DataFrame(results)


def run_engineer_pipeline(job: EngineerJob) -> DataSource:
    """Engineer features over ``job.origin`` and write partitions to ``job.output``."""
    origin_path = job.origin.resolve()
    output_path = job.output.resolve()
    os.makedirs(str(output_path), exist_ok=False)

    logger.info(f"Engineering {origin_path} to {output_path}")

    lf_whole = pl.scan_parquet(list(origin_path.glob("*.par*")))

    if job.dry_run:
        lf_whole = lf_whole.slice(0, 50)

    progress_bar = Progress(
        TextColumn("[bold blue] {task.description}", justify="left"),
        BarColumn(bar_width=40),
        TextColumn("[task.completed]{task.completed}/{task.total}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        speed_estimate_period=60.0 * 10,
    )
    progress_bar.start()
    n_rows = lf_whole.select(pl.len()).collect(engine="streaming").item()
    progress = progress_bar.add_task("Rows", total=n_rows)
    part_progress = progress_bar.add_task("Parts", total=job.n_partitions)
    rows_per_part = math.ceil(n_rows / job.n_partitions)

    lf_whole = lf_whole.with_row_index("idx").with_columns(
        (pl.col("idx") // rows_per_part).clip(0, job.n_partitions - 1).alias("part")
    )
    for i in range(job.n_partitions):
        lf = lf_whole.filter(pl.col("part") == i).drop(["idx", "part"])
        for col in job.len_cols:
            msg = lambda: logger.info(f"Calculating len of {col}, dtype: {dtype}")
            dtype = lf.schema[col]
            if dtype.is_(pl.Utf8):
                msg()
                lf = lf.with_columns(pl.col(col).str.len_chars().alias(f"{col}_len"))
            elif dtype.base_type().is_(pl.List):
                msg()
                lf = lf.with_columns(pl.col(col).list.len().alias(f"{col}_len"))
            else:
                logger.error(f"Len operation not supported for {dtype}")
                raise TypeError(f"Len operation not supported for {dtype}")

        if job.years_to_first:
            lf = lf.with_columns(
                counts_by_year_delta=(
                    pl.col("counts_by_year_years")
                    - pl.col("publication_date").dt.year()
                )
            ).with_columns(
                counts_by_year_delta_first=pl.col("counts_by_year_delta").list.first()
            )
            lf_cited_dates = (
                lf.explode("referenced_works")
                .group_by("referenced_works")
                .agg(
                    [
                        pl.col("publication_date").alias("cited_by_dates"),
                        pl.col("referenced_works").first().alias("id"),
                    ]
                )
            )
            lf = (
                lf.join(
                    lf_cited_dates,
                    on="id",
                    how="inner",
                )
                .with_columns(
                    missing=pl.col("cited_by_count")
                    - pl.col("cited_by_dates").list.len()
                )
                .with_columns(
                    pl.col("cited_by_dates")
                    .list.eval(pl.element().dt.year())
                    .alias("cited_by_years"),
                )
                .with_columns(
                    cited_by_delta_years=pl.col("cited_by_years")
                    - pl.col("publication_date").dt.year()
                )
                .with_columns(
                    cited_by_delta_years_first=pl.col(
                        "cited_by_delta_years"
                    ).list.min(),
                    cited_by_delta_days_first=pl.col("cited_by_dates").list.min()
                    - pl.col("publication_date"),
                )
            )

            if job.dry_run:
                lf = lf.collect()
                print(lf)
                os.makedirs("./temp", exist_ok=True)
                lf.write_json("./temp/engineer-dry-run.json")
                break

            output_fname = output_path / f"part_{i}.parquet"
            lf.sink_parquet(
                output_fname,
                statistics=True,
                compression="zstd",
                compression_level=1,
            )
            n_written = pl.scan_parquet(output_fname).select(pl.len()).collect(engine="streaming").item()
            progress_bar.update(progress, advance=n_written)
            progress_bar.update(part_progress, advance=1)

    return job.output
