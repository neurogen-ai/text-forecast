"""Local workstation runtime implementation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from builders import build_eval_example_progress, build_progress_bars
from data.pipeline.describe import DescribeJob, run_describe_pipeline
from data.pipeline.engineer import EngineerJob, run_engineer_pipeline
from data.preprocess.embed import get_embedder
from data.preprocess.pipeline import PreprocessJob, run_preprocess_pipeline
from data.sources.base import DataSource
from data.sources.local import LocalDataSource
from runtime.run_lifecycle import create_run
from training.pipeline.eval import EvalJob, run_eval_pipeline
from training.pipeline.train import TrainJob, run_train_pipeline

from .base import RunResult, Runtime, TextEmbedder
from config.loader import read_experiment_name


class LocalRuntime:
    """Local runtime: local filesystem sources and HuggingFace embedders."""

    supported_source_backends = {"local"}

    def get_source(self, root: str | Path, name: str) -> DataSource:
        # No existence check: output sources do not exist until written.
        return LocalDataSource(base_dir=Path(root), name=name)

    def get_embedder(self, key: str, **kwargs: Any) -> TextEmbedder:
        return get_embedder(key, **kwargs)

    def run_preprocess(self, job: PreprocessJob) -> DataSource:
        return run_preprocess_pipeline(job, self)

    def run_describe(self, job: DescribeJob) -> None:
        run_describe_pipeline(job)

    def run_engineer(self, job: EngineerJob) -> DataSource:
        return run_engineer_pipeline(job)

    def run_train(self, job: TrainJob) -> RunResult:
        """Create the run client-side (P6), then execute in-process."""
        mlflow_experiment = read_experiment_name(job.experiment_name)
        provisional_name = (
            job.run_name or job.run_suffix or "train"
        )
        run_id = create_run(
            env=job.env,
            experiment_name=mlflow_experiment,
            run_name=provisional_name,
            parent_id=job.parent_id,
        )
        job = replace(job, run_id=run_id)

        progress = None
        if job.progress:
            progress = build_progress_bars()
            for pb in progress:
                try:
                    pb.start()
                except RuntimeError:
                    # build_epoch_progress is already started.
                    pass
        return run_train_pipeline(job, progress=progress)

    def run_eval(self, job: EvalJob) -> RunResult:
        """Create the eval run client-side (P6), then execute in-process."""
        training_experiment = (
            read_experiment_name(job.module_name) if job.module_name else None
        )
        eval_experiment = job.experiment_name or (
            f"{training_experiment}-EVAL"
            if training_experiment is not None
            else f"eval-{job.run_id}"
        )
        eval_run_id = create_run(
            env=job.env,
            experiment_name=eval_experiment,
            run_name=f"{job.prefix}eval-{job.run_id}",
        )
        job = replace(job, eval_run_id=eval_run_id)

        progress = None
        if job.progress:
            progress = build_eval_example_progress()
            progress.start()
        return run_eval_pipeline(
            job,
            progress=(progress, progress, progress) if progress else None,
        )


Runtime.register(LocalRuntime)
