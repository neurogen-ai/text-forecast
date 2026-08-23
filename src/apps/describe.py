from __future__ import annotations

from datetime import datetime
from logging import getLogger
from pathlib import Path

import typer

from apps.source_args import (
    build_source_backend_from_cli,
    source_backend_arg,
    source_base_dir_arg,
    source_opt_arg,
    source_volume_arg,
)
from config.env import load_env
from data.pipeline.describe import DescribeJob
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
    runtime_name: str | None = typer.Option(
        None,
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

    source = source_backend_obj.get_source(dataset)

    job = DescribeJob(
        source=source,
        start_time=start_time,
        end_time=end_time,
        time_col=time_col,
        describe_cols=list(describe_cols),
        buckets=list(buckets),
        filter_expr=filter_expr if filter_expr else None,
    )
    runtime.run_describe(job)
