from __future__ import annotations

from datetime import datetime
from logging import getLogger
from pathlib import Path

import typer

from config.env import load_env
from runtime.factory import build_runtime
from training.pipeline.eval import EvalJob
from utils.logging import setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)

app = typer.Typer(pretty_exceptions_enable=False)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prefix: str = typer.Option("", "--prefix", help="Prefix to mlflow run name"),
    run_id: str = typer.Option(
        "", "--run-id", "-id", help="MLflow run ID to load model checkpoint from"
    ),
    epoch: int | None = typer.Option(
        None, "--epoch", "-e", help="Epoch to load checkpoint from"
    ),
    start_date: datetime | None = typer.Option(
        None, "--start-date", "-s", help="Datetime to start eval from"
    ),
    end_date: datetime | None = typer.Option(
        None, "--end-date", help="Datetime to end eval at"
    ),
    interval: int | None = typer.Option(
        None, "--interval", "-i", help="Interval to loop over successive time segments"
    ),
    interval_unit: str = typer.Option(
        "y", "--interval-unit", help="Unit of interval quantity"
    ),
    experiment: str | None = typer.Option(
        None, "--experiment", help="Eval MLflow experiment name override"
    ),
    runtime_name: str | None = typer.Option(
        None,
        "--runtime",
        help="Execution backend (local or modal); overrides [runtime].default",
    ),
    tracking_uri: str | None = typer.Option(
        None, "--tracking-uri", help="Override MLflow tracking URI"
    ),
    artifact_loc: Path | None = typer.Option(
        None, "--artifact-loc", help="Override artifact storage location"
    ),
    temp_dir: Path = typer.Option(
        "./.temp/", "--temp-dir", help="Folder to store temp data in"
    ),
    clean_up: bool = typer.Option(
        False, "--clean-up", help="Delete temporary files after run completes"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Dry run with subset of dataset"
    ),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Use GPU as device"),
) -> None:
    """Evaluate a run over sliding time windows using its experiment file."""
    assert run_id, "Provide a run id"
    assert epoch is not None, "Provide an epoch to load checkpoint from"
    assert start_date is not None, "Provide a start date"
    assert interval is not None, "Provide an interval"
    assert start_date or not interval, "Specify an interval with start_date"
    assert interval or not end_date, "Specify an end_date with interval"

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

    runtime = build_runtime(runtime_name, env)

    root_obj = ctx.find_root().obj

    job = EvalJob(
        run_id=run_id,
        epoch=epoch,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        interval_unit=interval_unit,
        prefix=prefix,
        module_name=root_obj["experiment_name"],
        experiment_name=experiment,
        dry_run=dry_run,
        gpu=gpu,
        temp_dir=str(temp_dir),
        clean_up=clean_up,
        env=env,
    )

    result = runtime.run_eval(job)
    logger.info(f"Eval run {result.run_id} {result.status}")


if __name__ == "__main__":
    app()
