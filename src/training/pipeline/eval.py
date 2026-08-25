"""Eval job spec and venue-independent execution."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import mlflow
import torch
from torch.utils.data import DataLoader

from config.env import Env
from config.loader import load_experiment_from_path
from data.datasets.graph_dataset import GraphDataset
from runtime.base import RunResult
from training.checkpointing import CheckpointRef
from training.checkpointing.mlflow_store import MlflowCheckpointProcessor
from training.engine import Engine, ProgressBars
from training.pipeline._common import build_run_context


class _DateTimeVals(NamedTuple):
    year: int
    month: int
    day: int


_T_DELTA_MAP = {
    "y": lambda interval: _DateTimeVals(year=interval, month=0, day=0),
}


@dataclass(frozen=True, kw_only=True)
class EvalJob:
    """Everything needed to evaluate one training run over time windows.

    ``experiment_name`` is the *training* run's MLflow experiment (needed to
    locate its artifacts); the eval run lands in
    ``<experiment_name>-EVAL`` unless ``eval_experiment_name`` overrides it.
    The eval run itself is created client-side by the runtime wrapper and its
    id passed as ``eval_run_id``.
    """

    run_id: str                 # training run to evaluate
    epoch: int
    start_date: datetime
    end_date: datetime | None = None
    interval: int | None = None
    interval_unit: str = "y"
    prefix: str = ""
    experiment_name: str | None = None  # override for the eval MLflow experiment
    eval_run_id: str = ""
    dry_run: bool = False
    gpu: bool = True
    temp_dir: str = "./.temp/"
    clean_up: bool = False
    env: Env


def _validate(job: EvalJob) -> _DateTimeVals:
    assert job.interval is not None, "Provide an interval"
    assert job.interval_unit in _T_DELTA_MAP, (
        f"Interval unit must be one of: {list(_T_DELTA_MAP.keys())}"
    )
    return _T_DELTA_MAP[job.interval_unit](job.interval)


def run_eval_pipeline(
    job: EvalJob,
    *,
    progress: ProgressBars | None = None,
) -> RunResult:
    """Evaluate a checkpoint over sliding time windows.

    Identical on local and Modal.  Prediction exports are written to a local
    temp dir per window and logged to the eval run as ``exports/<year>``
    MLflow artifacts — the export channel on every venue.
    """
    if not job.eval_run_id:
        raise ValueError(
            "EvalJob.eval_run_id is empty: create the eval MLflow run "
            "client-side and replace() it onto the job before dispatching"
        )
    t_delta = _validate(job)

    # "<training-experiment>-EVAL" when the training experiment name is known;
    # otherwise fall back to a run-derived name.
    mlflow_experiment = (
        f"{job.experiment_name}-EVAL"
        if job.experiment_name is not None
        else f"eval-{job.run_id}"
    )

    temp_root = Path(job.temp_dir)
    checkpoint_dir = temp_root / job.run_id / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    processor = MlflowCheckpointProcessor(
        artifact_loc=job.env.artifact_loc,
        tracking_uri=job.env.tracking_uri,
        experiment_name=job.experiment_name or mlflow_experiment,
    )

    if not processor.experiment_file_exists(run_id=job.run_id):
        raise ValueError(f"No experiment file found for run {job.run_id}")

    runtime = build_run_context(
        gpu=job.gpu,
        subsample=512 if job.dry_run else None,
    )

    experiment_file_path = processor.download_experiment_file(
        run_id=job.run_id, dest=checkpoint_dir
    )
    exp = load_experiment_from_path(experiment_file_path, runtime, env=job.env)

    checkpoint = processor.load(
        ref=CheckpointRef(run_id=job.run_id, epoch=job.epoch),
        map_location=runtime.device,
    )
    exp.model.load_state_dict(checkpoint.model)
    exp.model.eval()
    if runtime.device.type == "cuda":
        exp.model.compile(mode="max-autotune")

    engine = Engine(experiment=exp, runtime=runtime, progress=progress)

    export_root = checkpoint_dir.parent / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    base_dataset = exp.val_loader.dataset
    if not isinstance(base_dataset, GraphDataset):
        raise TypeError("Eval windows require a GraphDataset")

    last_metrics: dict[str, float] = {}
    current_t_start = job.start_date

    with mlflow.start_run(run_id=job.eval_run_id):
        mlflow.set_tracking_uri(job.env.tracking_uri)
        mlflow.log_params(
            {
                "eval.source_run_id": job.run_id,
                "eval.epoch": job.epoch,
                "eval.prefix": job.prefix,
                "eval.start_date": job.start_date.isoformat(),
                "eval.end_date": (
                    job.end_date.isoformat() if job.end_date else ""
                ),
                "eval.interval": job.interval,
                "eval.dry_run": job.dry_run,
            }
        )

        while True:
            next_t = datetime(
                current_t_start.year + t_delta.year,
                current_t_start.month + t_delta.month,
                current_t_start.day + t_delta.day,
            )
            t_end = (
                min(next_t, job.end_date)
                if job.end_date is not None
                else next_t
            )

            windowed = base_dataset.with_window(
                t_start=current_t_start.date(),
                t_end=t_end.date(),
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

            export_dir = export_root / f"{current_t_start.year}"
            exp.tracker.export = True
            exp.tracker.export_loc = export_dir

            engine.eval_epoch(epoch=current_t_start.year, loader=window_loader)
            metrics = exp.tracker.report(
                progress_bar=(
                    progress[0] if progress is not None else None
                ),
                epoch=current_t_start.year,
            )
            if metrics:
                mlflow.log_metrics(
                    metrics,
                    step=current_t_start.year,
                    timestamp=current_t_start.year,
                    synchronous=False,
                    run_id=job.eval_run_id,
                )
            mlflow.log_metric(
                "examples",
                len(windowed),
                step=current_t_start.year,
                run_id=job.eval_run_id,
            )
            mlflow.log_artifacts(
                str(export_dir), artifact_path=f"exports/{current_t_start.year}"
            )
            exp.tracker.clear()
            last_metrics = dict(metrics) if metrics else {}

            current_t_start = t_end
            if job.end_date is not None and current_t_start >= job.end_date:
                break

    if job.clean_up:
        shutil.rmtree(temp_root / job.run_id, ignore_errors=True)

    return RunResult(
        run_id=job.eval_run_id,
        status="completed",
        metrics=last_metrics,
    )
