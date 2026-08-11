"""Runtime-agnostic preprocessing pipeline."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any

import polars as pl
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from data.preprocess import clean_step, tokenise_step
from data.sources.base import DataSource
from runtime.base import Runtime, TextEmbedder

logger = getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class PreprocessJob:
    """Fully serialisable description of a preprocessing job.

    The job contains no runtime state; it is passed to
    ``Runtime.run_preprocess`` which decides whether to execute locally or
    dispatch to a remote container.
    """

    origin: DataSource
    destination: DataSource
    clean_cols: list[str] = field(default_factory=list)
    clean_levels: list[int] = field(default_factory=list)
    clean_min_len: list[int] = field(default_factory=list)
    tokeniser: str = ""
    tokenise_cols: list[str] = field(default_factory=list)
    embedder_key: str = ""
    embedder_kwargs: dict[str, Any] = field(default_factory=dict)
    embed_cols: list[str] = field(default_factory=list)
    n_partitions: int = 0
    rows_per_part: int = 0
    compression_level: int = 1
    start_date: datetime | None = None
    end_date: datetime | None = None
    field_id: list[int] = field(default_factory=list)
    drop_na_cols: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    filt_license: bool = True
    replace_non_permissive_cols: list[str] = field(default_factory=list)
    dry_run: bool = False
    max_threads: int = 8
    runtime_name: str = "local"
    params: dict[str, Any] = field(default_factory=dict)


def _measure_lf(lf: pl.LazyFrame, run: bool = False) -> float:
    if run:
        return lf.select(pl.len()).collect(engine="streaming").item()
    return float("nan")


def run_preprocess_pipeline(job: PreprocessJob, runtime: Runtime) -> DataSource:
    """Execute the full preprocessing job using the supplied runtime."""
    os.environ["POLARS_MAX_THREADS"] = f"{job.max_threads}"

    source_path = runtime.maybe_download(job.origin)
    destination_path = job.destination.resolve()
    os.makedirs(destination_path, exist_ok=True)
    logger.info(f"Preprocessing {source_path} -> {destination_path}")

    lf_whole: pl.LazyFrame = pl.scan_parquet(
        list(source_path.glob("*.par*")), extra_columns="ignore"
    )

    if job.filt_license:
        lf_whole = lf_whole.filter(pl.col("is_license_safe"))
    else:
        logger.warning("Including non-permissive licenses")

    if job.start_date is not None:
        lf_whole = lf_whole.filter(pl.col("publication_date") >= job.start_date)
    else:
        logger.warning("No start date set")

    if job.end_date is not None:
        lf_whole = lf_whole.filter(pl.col("publication_date") < job.end_date)
    else:
        logger.warning("No end date set")

    if job.field_id:
        lf_whole = lf_whole.filter(pl.col("field_id").is_in(job.field_id))
    else:
        logger.warning("No field id filter set")

    if job.languages:
        lf_whole = lf_whole.filter(pl.col("language").is_in(job.languages))
    else:
        logger.warning("No Language filters set")

    if job.types:
        lf_whole = lf_whole.filter(pl.col("type").is_in(job.types))
    else:
        logger.warning("No document type filters set")

    n_rows: float = _measure_lf(lf_whole, True)
    rows_per_part = job.rows_per_part
    n_partitions = job.n_partitions
    if n_partitions:
        rows_per_part = math.ceil(n_rows / n_partitions)
    elif rows_per_part:
        n_partitions = math.ceil(n_rows / rows_per_part)
    else:
        n_partitions = 1
        rows_per_part = max(1, math.ceil(n_rows))

    lf_whole = lf_whole.with_row_index("idx").with_columns(
        (pl.col("idx") // rows_per_part).clip(0, n_partitions - 1).alias("part")
    )

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
    progress = progress_bar.add_task("Rows", total=int(n_rows))

    if job.dry_run:
        lf_whole = lf_whole.slice(0, 500)

    embedder_instance: TextEmbedder | None = None
    if job.embedder_key:
        embedder_instance = runtime.get_embedder(
            key=job.embedder_key, **job.embedder_kwargs
        )

    def embed_batch(series: pl.Series) -> pl.Series:
        assert embedder_instance is not None
        texts = [t if isinstance(t, str) else "" for t in series.to_list()]
        vectors = embedder_instance.encode(texts)
        return pl.Series(
            vectors,
            dtype=pl.Array(pl.Float32, width=embedder_instance.output_dim),
        )

    for i in range(n_partitions):
        lf = lf_whole.filter(pl.col("part") == i).drop(["idx", "part"])

        for col in job.replace_non_permissive_cols:
            logger.info(f"Replacing non-permissive {col} with None")
            lf = lf.with_columns(
                pl.when(pl.col("is_license_safe") == False)
                .then(pl.lit(None, dtype=lf.schema[col]))
                .otherwise(pl.col(col))
                .alias(col)
            )

        if job.clean_cols:
            for col, level, min_len in zip(
                job.clean_cols, job.clean_levels, job.clean_min_len
            ):
                l1 = _measure_lf(lf)
                lf = clean_step(lf=lf, col=col, min_len=min_len, level=level)
                l2 = _measure_lf(lf)
                logger.info(f"Dropped {l1 - l2:,.0f} in {col} at clean lvl {level}")
        elif i == 0:
            logger.warning("No cols selected for cleaning")

        if job.drop_na_cols:
            for col in job.drop_na_cols:
                lf = lf.drop_nulls(subset=col)
        elif i == 0:
            logger.warning("No columns set to drop nulls")

        if embedder_instance:
            if i == 0:
                logger.info(
                    f"Embedding {', '.join(job.embed_cols)} with {job.embedder_key}"
                )
            lf = (
                lf.drop_nulls(job.embed_cols)
                .with_columns(
                    to_embed=pl.concat_str(
                        [pl.col(col) for col in job.embed_cols],
                        separator=" ",
                    )
                )
                .with_columns(
                    pl.col("to_embed")
                    .map_batches(
                        embed_batch,
                        return_dtype=pl.Array(
                            pl.Float32, width=embedder_instance.output_dim
                        ),
                    )
                    .alias(f"{' '.join(job.embed_cols)}_embedding")
                )
                .drop("to_embed")
            )

        if job.tokeniser:
            if i == 0:
                logger.info(
                    f"Tokenising {', '.join(job.tokenise_cols)} with {job.tokeniser}"
                )
            lf = tokenise_step(
                lf=lf,
                tokeniser_path=job.tokeniser,
                columns=job.tokenise_cols,
            )

        if job.dry_run:
            print(lf.collect())
            break

        output_fname = destination_path / f"part_{i}.parquet"
        lf.sink_parquet(
            output_fname,
            statistics=True,
            compression="zstd",
            compression_level=job.compression_level,
        )
        n_written = (
            pl.scan_parquet(output_fname)
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )
        progress_bar.update(progress, advance=n_written)

    metadata = {k: str(v) for k, v in job.params.items()}
    metadata["runtime"] = job.runtime_name
    metadata["embedder"] = job.embedder_key
    metadata["embedder_kwargs"] = job.embedder_kwargs
    with open(destination_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    runtime.maybe_upload(job.destination, destination_path)
    return job.destination
