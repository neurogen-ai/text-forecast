"""Runtime-agnostic preprocessing pipeline."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, fields
from datetime import datetime
from logging import getLogger
from typing import Any, TYPE_CHECKING

import polars as pl
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from collections.abc import Callable

from data.preprocess.clean import run_clean
from data.preprocess.embed_huggingface import EMBEDDERS
from data.preprocess.steps import (
    CleanStep,
    DropNaStep,
    EmbedStep,
    PipelineStep,
    StepContext,
    TokeniseStep,
    step_from_dict,
    step_to_dict,
)
from data.preprocess.tokenise import main as tokenise_step
from data.sources.base import DataSource
from utils.logging import setup_logger
if TYPE_CHECKING:
    from runtime.base import Runtime, TextEmbedder

logger = getLogger(__name__)
_ = setup_logger(logger)

@dataclass(frozen=True, kw_only=True)
class PreprocessJob:
    """Fully serialisable description of a preprocessing job.

    The job contains no runtime state; it is passed to
    ``Runtime.run_preprocess`` which decides whether to execute locally or
    dispatch to a remote container. All per-column operations are described
    declaratively via ``steps`` (see :mod:`data.preprocess.steps`).
    """

    origin: DataSource
    destination: DataSource
    steps: tuple[PipelineStep, ...] = ()
    n_partitions: int = 0
    rows_per_part: int = 0
    compression_level: int = 1
    start_date: datetime | None = None
    end_date: datetime | None = None
    field_id: list[int] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    filt_license: bool = True
    replace_non_permissive_cols: list[str] = field(default_factory=list)
    dry_run: bool = False
    max_threads: int = 8
    runtime_name: str = "local"
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict; DataSource fields stay as objects."""
        out: dict[str, Any] = {
            f.name: getattr(self, f.name) for f in fields(self)
        }
        out["steps"] = [step_to_dict(s) for s in self.steps]
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreprocessJob:
        """Reconstruct a job; steps are rebuilt via ``step_from_dict``."""
        kwargs = {k: v for k, v in d.items() if k != "steps"}
        steps = tuple(step_from_dict(s) for s in d.get("steps", ()))
        return cls(steps=steps, **kwargs)


def _run_drop_na(
    lf: pl.LazyFrame,
    step: DropNaStep,
    ctx: StepContext | None = None,  # noqa: ARG001 — uniform handler signature
) -> pl.LazyFrame:
    del ctx
    return lf.drop_nulls(subset=list(step.cols))


def _run_tokenise(
    lf: pl.LazyFrame,
    step: TokeniseStep,
    ctx: StepContext | None = None,  # noqa: ARG001 — uniform handler signature
) -> pl.LazyFrame:
    del ctx
    return tokenise_step(
        lf=lf,
        tokeniser_path=step.tokeniser,
        columns=list(step.cols),
    )


def _run_embed(
    lf: pl.LazyFrame,
    step: EmbedStep,
    ctx: StepContext,
) -> pl.LazyFrame:
    spec = EMBEDDERS[step.embedder_key]
    embedder = ctx.embedder
    assert embedder is not None  # guaranteed by StepContext construction

    def embed_batch(series: pl.Series) -> pl.Series:
        texts = [t if isinstance(t, str) else "" for t in series.to_list()]
        vectors = embedder.encode(texts)
        return pl.Series(
            vectors,
            dtype=pl.Array(pl.Float32, width=embedder.output_dim),
        )

    return (
        lf.drop_nulls(step.cols)
        .with_columns(
            [
                pl.lit(spec.bos_token) + pl.col(col) + pl.lit(spec.eos_token)
                for col in step.cols
            ]
        )
        .with_columns(
            to_embed=pl.concat_str(
                [pl.col(col) for col in step.cols],
                separator="",
            )
        )
        .with_columns(
            pl.col("to_embed")
            .map_batches(
                embed_batch,
                return_dtype=pl.Array(
                    pl.Float32, width=embedder.output_dim
                ),
            )
            .alias(f"{'_'.join(step.cols)}_embedding")
        )
        .drop("to_embed")
    )


HANDLERS: dict[type[PipelineStep], Callable[..., pl.LazyFrame]] = {
    CleanStep: run_clean,
    DropNaStep: _run_drop_na,
    TokeniseStep: _run_tokenise,
    EmbedStep: _run_embed,
}


def _measure_lf(lf: pl.LazyFrame, run: bool = False) -> float:
    if run:
        return lf.select(pl.len()).collect(engine="streaming").item()
    return float("nan")


def run_preprocess_pipeline(job: PreprocessJob, runtime: Runtime) -> DataSource:
    """Execute the full preprocessing job using the supplied runtime."""
    os.environ["POLARS_MAX_THREADS"] = f"{job.max_threads}"

    source_path = job.origin.resolve()
    destination_path = job.destination.resolve()
    os.makedirs(destination_path, exist_ok=True)
    logger.info(f"Preprocessing {source_path} -> {destination_path}")
    files = list(source_path.glob("*.par*"))
    lf_whole: pl.LazyFrame = pl.scan_parquet(
        files, extra_columns="ignore"
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
        n_partitions = len(files)
        rows_per_part = max(1, math.ceil(n_rows / n_partitions))

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

    embed_step = next((s for s in job.steps if isinstance(s, EmbedStep)), None)
    embedder_instance: TextEmbedder | None = None
    if embed_step is not None:
        embedder_instance = runtime.get_embedder(
            key=embed_step.embedder_key, **embed_step.embedder_kwargs
        )

    for i in range(n_partitions):
        logger.debug(f'Beggining part {i}')
        lf = lf_whole.filter(pl.col("part") == i).drop(["idx", "part"])

        for col in job.replace_non_permissive_cols:
            logger.info(f"Replacing non-permissive {col} with None")
            lf = lf.with_columns(
                pl.when(pl.col("is_license_safe") == False)
                .then(pl.lit(None, dtype=lf.schema[col]))
                .otherwise(pl.col(col))
                .alias(col)
            )

        ctx = StepContext(
            runtime=runtime,
            embedder=embedder_instance,
            partition_index=i,
            logger=logger,
        )
        for step in job.steps:
            if isinstance(step, (CleanStep, DropNaStep)):
                rows_before = _measure_lf(lf, True)
                lf = HANDLERS[type(step)](lf, step, ctx)
                rows_after = _measure_lf(lf, True)
                delta = f"{type(step).__name__} row delta {rows_after - rows_before:,.0f}"
                logger.info(delta)
            else:
                lf = HANDLERS[type(step)](lf, step, ctx)

        if job.dry_run:
            print(lf.collect())
            break

        output_fname = destination_path / f"part_{i}.parquet"
        logger.info(f'Writing {output_fname}')
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

    metadata: dict[str, Any] = {k: str(v) for k, v in job.params.items()}
    metadata["runtime"] = job.runtime_name
    if embed_step is not None:
        metadata["embedder"] = embed_step.embedder_key
        metadata["embedder_kwargs"] = embed_step.embedder_kwargs
    with open(destination_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return job.destination
