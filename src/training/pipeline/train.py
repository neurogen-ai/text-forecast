"""Train job spec and venue-independent execution."""

from __future__ import annotations

import mlflow
import torch
from mlflow.tracking import MlflowClient

from config.env import Env
from config.loader import load_experiment_from_path
from runtime.base import RunResult
from runtime.run_lifecycle import set_run_name
from training.checkpointing import CheckpointRef
from training.checkpointing.base import ExperimentFileStore
from training.engine import Engine, ProgressBars
from training.pipeline._common import (
    build_run_context,
    experiment_file,
    model_source_file,
)

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class TrainJob:
    """Everything needed to run one training schedule.

    ``run_id`` is filled in by the runtime wrapper (which owns the client-side
    run lifecycle) via ``dataclasses.replace``; the pipeline resumes that run.
    """

    experiment_name: str       # module name (root CLI/toml resolution)
    experiment_source: bytes   # raw experiment file bytes
    run_name: str = ""
    run_suffix: str = ""      # composed with the model class name in-pipeline
    parent_id: str | None = None
    start_epoch: int | None = None
    compile_mode: str = ""
    fullgraph: bool = False
    load_id: str = ""
    load_epoch: int | None = None
    model_only: bool = False   # skip optimizer/scheduler restore on resume
    subsample: int | None = None
    gpu: bool = True
    progress: bool = True
    env: Env
    run_id: str = ""


def _resolve_start_epoch(job: TrainJob) -> int:
    if job.start_epoch is not None:
        return job.start_epoch
    if job.load_epoch is not None:
        return job.load_epoch + 1
    return 1


def run_train_pipeline(
    job: TrainJob,
    *,
    progress: ProgressBars | None = None,
) -> RunResult:
    """Execute a training schedule against a pre-created MLflow run.

    Identical on local and Modal.  The caller (a ``Runtime`` wrapper) has
    already created the run client-side and stamped its id onto the job.
    """
    if not job.run_id:
        raise ValueError(
            "TrainJob.run_id is empty: create the MLflow run client-side "
            "and replace() it onto the job before dispatching"
        )

    with experiment_file(job.experiment_source) as path:
        runtime = build_run_context(
            gpu=job.gpu,
            compile_mode=job.compile_mode,
            fullgraph=job.fullgraph,
            subsample=job.subsample,
        )
        exp = load_experiment_from_path(path, runtime, env=job.env)

        if runtime.compile_mode:
            exp.model.compile(
                mode=runtime.compile_mode, fullgraph=runtime.fullgraph
            )

        if job.load_id and job.load_epoch is not None:
            checkpoint = exp.checkpoints.load(
                ref=CheckpointRef(run_id=job.load_id, epoch=job.load_epoch),
                map_location=runtime.device,
            )
            exp.model.load_state_dict(checkpoint.model)
            if not job.model_only:
                exp.strategy.load_optimizer_state(
                    optimizer=checkpoint.optimizer,
                    scheduler=checkpoint.scheduler,
                )

        mlflow.set_tracking_uri(job.env.tracking_uri)

        with mlflow.start_run(run_id=job.run_id):
            model_name = type(exp.model).__name__
            if job.run_name:
                final_name = job.run_name
            else:
                final_name = f"{model_name}-{job.run_suffix}"
                if job.subsample:
                    final_name += "-DRY"
            set_run_name(job.run_id, final_name)

            if isinstance(exp.checkpoints, ExperimentFileStore):
                exp.checkpoints.save_experiment_file(path=path)

            model_file = model_source_file(exp.model)
            if model_file is not None and model_file.exists():
                mlflow.log_artifact(str(model_file))

            mlflow.log_params(
                {
                    "experiment_name": exp.experiment_name,
                    "model.class": model_name,
                    "train.examples": len(exp.train_loader.dataset),
                    "val.examples": len(exp.val_loader.dataset),
                }
            )

            engine = Engine(experiment=exp, runtime=runtime, progress=progress)
            metrics = engine.fit(
                start_epoch=_resolve_start_epoch(job), run_id=job.run_id
            )

        artifacts = MlflowClient().list_artifacts(job.run_id)
        checkpoints = [
            a.path for a in artifacts if a.path.endswith(".pt")
        ]

    return RunResult(
        run_id=job.run_id,
        status="completed",
        checkpoints=checkpoints,
        metrics=dict(metrics) if metrics else {},
    )
