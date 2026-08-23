from __future__ import annotations

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
from data.pipeline.engineer import EngineerJob
from data.sources import DataSource, LocalDataSource, SourceBackend
from runtime import build_runtime
from utils.logging import setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)
app = typer.Typer(pretty_exceptions_enable=False)


def _resolve_output_source(
    output_value: str,
    input_path: Path | None,
    input_dataset: str | None,
    source_backend: SourceBackend,
) -> DataSource:
    """Resolve --output as explicit path or dataset name via source backend."""
    if not output_value:
        if input_path is not None:
            return LocalDataSource(
                base_dir=input_path.parent,
                name=f"{input_path.name}-engineered",
            )
        if input_dataset:
            return source_backend.get_source(f"{input_dataset}-engineered")
        raise ValueError("Cannot derive output without input_path or input_dataset")

    candidate = Path(output_value)
    if candidate.is_absolute() or "/" in output_value:
        return LocalDataSource(base_dir=candidate.parent, name=candidate.name)
    return source_backend.get_source(output_value)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input_path: Path = typer.Option(
        "",
        "--input-path",
        "-i",
        help="Path to input data (local filesystem only)",
    ),
    input_dataset: str = typer.Option(
        "", "--input-dataset", "-d", help="Dataset name to use as input"
    ),
    output_path: str = typer.Option(
        "",
        "--output",
        "-o",
        help='Explicit output path or dataset name; defaults to "<input>-engineered"',
    ),
    len_cols: list[str] = typer.Option(
        [],
        "--calc-len",
        "-l",
        help='Columns to calculate the len of, placed in "colname_len"',
    ),
    years_to_first: bool = typer.Option(
        True,
        "--years-to-first",
        help="Calulate years to first citation -> colname: years_to_first_citation",
    ),
    n_partitions: int = typer.Option(
        64,
        help="No. of partitions to slice data into",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Test/Dry run with 500 sample slice of data",
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
    assert input_path or input_dataset, "Provide path to or name of an input dataset"

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

    if input_path:
        origin: DataSource = LocalDataSource(
            base_dir=input_path.parent, name=input_path.name
        )
    else:
        origin = source_backend_obj.get_source(input_dataset)

    output = _resolve_output_source(
        output_value=output_path,
        input_path=input_path or None,
        input_dataset=input_dataset or None,
        source_backend=source_backend_obj,
    )

    job = EngineerJob(
        origin=origin,
        output=output,
        len_cols=list(len_cols),
        years_to_first=years_to_first,
        n_partitions=n_partitions,
        dry_run=dry_run,
    )
    runtime.run_engineer(job)
