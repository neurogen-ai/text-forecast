from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import NamedTuple

import mlflow
import torch
import torch._logging
import typer
from torch.utils.data import DataLoader

from builders import build_eval_example_progress
from config.env import load_env
from config.loader import load_experiment_from_path
from config.runtime import RunContext
from data.datasets.graph_dataset import GraphDataset
from training.checkpointing import CheckpointRef
from training.checkpointing.mlflow_store import MlflowCheckpointProcessor
from training.engine import Engine
from utils.logging import setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)

app = typer.Typer(pretty_exceptions_enable=False)


class DateTimeVals(NamedTuple):
    year: int
    month: int
    day: int


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
    experiment: str = typer.Option(
        "", "--experiment", help="MLflow experiment name override"
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

    base_dir = Path(env.source.get("local", {}).get("base_dir", "/tmp/data"))
    PREDICTIONS_DIR = base_dir / "eval" / run_id / f"run-{prefix}"

    device = torch.device(
        "cuda" if torch.cuda.is_available() and gpu else "cpu"
    )
    assert device.type == "cuda" or not gpu, (
        "No GPU available on this device, use --no-gpu option"
    )

    runtime = RunContext(
        device=device,
        dtype=torch.float32,
        subsample=512 if dry_run else None,
    )

    root_obj = ctx.find_root().obj
    mlflow_experiment = experiment or f"{root_obj['experiment_name']}-EVAL"
    if PREDICTIONS_DIR.exists():
        logger.warning(f"Removing data in {PREDICTIONS_DIR}")
        time.sleep(5)
        shutil.rmtree(PREDICTIONS_DIR)
    os.makedirs(PREDICTIONS_DIR)

    torch._logging.set_logs(all=logging.ERROR)
    logger.info(f"MLflow experiment: {mlflow_experiment}")

    mlflow.set_tracking_uri(env.tracking_uri)
    mlflow.set_experiment(mlflow_experiment)
    logger.info(f"MLflow tracking URI connected: {env.tracking_uri}")

    CHECKPOINT_DIR = temp_dir / run_id / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    processor = MlflowCheckpointProcessor(
        artifact_loc=env.artifact_loc,
        tracking_uri=env.tracking_uri,
        experiment_name=mlflow_experiment,
    )

    if not processor.experiment_file_exists(run_id=run_id):
        raise typer.BadParameter(
            f"No experiment file found for run {run_id}"
        )

    experiment_file_path = processor.download_experiment_file(
        run_id=run_id, dest=CHECKPOINT_DIR
    )
    exp = load_experiment_from_path(experiment_file_path, runtime, env=env)

    checkpoint = processor.load(
        ref=CheckpointRef(run_id=run_id, epoch=epoch),
        map_location=device,
    )
    exp.model.load_state_dict(checkpoint.model)
    exp.model.eval()
    if device.type == "cuda":
        exp.model.compile(mode="max-autotune")

    logger.info(f"Model {exp.model.__module__} loaded to {device}")

    example_progress_bar = build_eval_example_progress(disable=False)
    example_progress_bar.start()
    engine = Engine(
        experiment=exp,
        runtime=runtime,
        progress=(
            example_progress_bar,
            example_progress_bar,
            example_progress_bar,
        ),
    )

    t_delta_map = {"y": DateTimeVals(year=interval, month=0, day=0)}
    assert interval_unit in t_delta_map, (
        f"Interval unit must be one of: {list(t_delta_map.keys())}"
    )
    T_DELTA = t_delta_map[interval_unit]

    current_t_start = start_date
    base_dataset = exp.val_loader.dataset
    if not isinstance(base_dataset, GraphDataset):
        raise TypeError("Eval windows require a GraphDataset")

    with mlflow.start_run(
        run_name=f"{prefix}-{type(exp.model).__name__}-{run_id}"
    ):
        mlflow.log_params(ctx.params)

        while True:
            current_t_end = datetime(
                current_t_start.year + T_DELTA.year,
                current_t_start.month + T_DELTA.month,
                current_t_start.day + T_DELTA.day,
            )
            if end_date is not None and current_t_end > end_date:
                current_t_end = end_date

            windowed = base_dataset.with_window(
                t_start=current_t_start.date(),
                t_end=current_t_end.date(),
            )
            window_loader = DataLoader(
                dataset=windowed,
                batch_size=exp.val_loader.batch_size,
                num_workers=exp.val_loader.num_workers,
                prefetch_factor=exp.val_loader.prefetch_factor,
                persistent_workers=exp.val_loader.persistent_workers,
                pin_memory=exp.val_loader.pin_memory,
                shuffle=False,
                drop_last=exp.val_loader.drop_last,
            )

            exp.tracker.export = True
            exp.tracker.export_loc = PREDICTIONS_DIR / f"{current_t_start.year}"

            engine.eval_epoch(epoch=current_t_start.year, loader=window_loader)
            metrics = exp.tracker.report(
                progress_bar=example_progress_bar,
                epoch=current_t_start.year,
            )
            if metrics:
                mlflow.log_metrics(
                    metrics,
                    step=current_t_start.year,
                    timestamp=current_t_start.year,
                    synchronous=False,
                )
            mlflow.log_metric(
                "examples", len(windowed), step=current_t_start.year
            )
            exp.tracker.clear()

            current_t_start = current_t_end
            if end_date is not None and current_t_start >= end_date:
                break

    if clean_up:
        shutil.rmtree(temp_dir / run_id, ignore_errors=True)

    logger.info("Eval Finished")


if __name__ == "__main__":
    app()
