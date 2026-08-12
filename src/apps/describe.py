from __future__ import annotations

from datetime import datetime
from logging import getLogger
from pathlib import Path

import polars as pl
import typer

from apps.source_args import (
    build_source_backend_from_cli,
    source_backend_arg,
    source_base_dir_arg,
    source_opt_arg,
    source_volume_arg,
)
from config.env import load_env
from runtime import build_runtime
from utils.logging import setup_logger


logger = getLogger(__name__)
_ = setup_logger(logger)

app = typer.Typer(pretty_exceptions_enable=False)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    dataset: str = typer.Option("", "--dataset", "-d", help="Dataset name"),
    start_time: datetime | None = typer.Option(
        None, "--start-time", "-s", help="Start date/time to filter by, inclusive"
    ),
    end_time: datetime | None = typer.Option(
        None, "--end-time", "-e", help="End date/time to filter by, exclusive"
    ),
    time_col: str = typer.Option(
        "publication_date", "--time-col", "-tc", help="Date/Time col in dataset."
    ),
    describe_cols: list[str] = typer.Option(
        [], "--column", "-c", help="Columns to describe"
    ),
    buckets: list[float] = typer.Option(
        [],
        "--bucket",
        "-b",
        help="Borders of buckets to describe count by, creates list of inclusive upper limits",
    ),
    filter_expr: str = typer.Option(
        "",
        "--filter-expr",
        help="Raw Polars expression string applied to the dataset",
    ),
    min_cited_by_count: int | None = typer.Option(
        None,
        "--min-cited-by-count",
        help="Minimum cited_by_count filter",
    ),
    min_referenced_works: int | None = typer.Option(
        None,
        "--min-referenced-works",
        help="Minimum referenced_works list length filter",
    ),
    runtime_name: str = typer.Option(
        "local",
        "--runtime",
        "-r",
        help="Execution backend (local or modal)",
    ),
    source_backend: str | None = source_backend_arg(),
    source_opts: list[str] = source_opt_arg(),
    source_base_dir: Path | None = source_base_dir_arg(),
    source_volume: str | None = source_volume_arg(),
    tracking_uri: str | None = typer.Option(
        None, "--tracking-uri", help="Override MLflow tracking URI"
    ),
    artifact_loc: Path | None = typer.Option(
        None, "--artifact-loc", help="Override artifact storage location"
    ),
) -> None:
    env = load_env(
        overrides={
            k: v
            for k, v in {
                "tracking_uri": tracking_uri,
                "artifact_loc": artifact_loc,
            }.items()
            if v is not None
        }
    )

    source_backend_obj = build_source_backend_from_cli(
        env=env,
        source_backend=source_backend,
        source_opts=source_opts,
        source_base_dir=source_base_dir,
        source_volume=source_volume,
    )

    if runtime_name is None:
        runtime_name = env.runtime.get("default", "local")
    runtime = build_runtime(runtime_name, env)

    supported = getattr(runtime, "supported_source_backends", {"local"})
    if source_backend_obj.name not in supported:
        raise typer.BadParameter(
            f"Runtime {runtime_name!r} does not support source backend "
            f"{source_backend_obj.name!r}"
        )

    if runtime_name == "modal":
        raise typer.BadParameter(
            "Modal runtime is not yet supported for describe. "
            "Use --runtime local with a Modal volume mounted locally, "
            "or choose --source-backend local."
        )

    source = source_backend_obj.get_source(dataset)
    source_path = source.resolve()
    logger.info(f"Describing dataset at {source_path} (backend={source_backend_obj.name})")

    lf = pl.scan_parquet(list(source_path.glob("*.par*")))

    if start_time is not None:
        lf = lf.filter(pl.col(time_col) >= start_time)
    if end_time is not None:
        lf = lf.filter(pl.col(time_col) < end_time)

    if filter_expr:
        lf = lf.filter(eval(filter_expr))
    else:
        exprs: list[pl.Expr] = []
        if min_cited_by_count is not None:
            exprs.append(pl.col("cited_by_count") >= min_cited_by_count)
        if min_referenced_works is not None:
            exprs.append(pl.col("referenced_works").list.len() >= min_referenced_works)
        if exprs:
            lf = lf.filter(pl.all_horizontal(exprs))

    n_buckets = len(buckets)
    for col in describe_cols:
        bucket_counts = {
            f"{buckets[i]}-{buckets[i + 1]}": None for i in range(n_buckets - 1)
        }
        weights = {k: v for k, v in bucket_counts.items()}
        for i in range(n_buckets - 1):
            bucket_counts[f"{buckets[i]}-{buckets[i + 1]}"] = (
                lf.filter((pl.col(col) >= buckets[i]) & (pl.col(col) < buckets[i + 1]))
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
