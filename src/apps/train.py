from __future__ import annotations

import os
import warnings
from logging import getLogger
from pathlib import Path


import mlflow
import torch
import typer

from apps.source_args import (
    build_source_backend_from_cli,
    source_backend_arg,
    source_base_dir_arg,
    source_opt_arg,
    source_volume_arg,
)
from builders import build_progress_bars
from config.env import load_env
from config.loader import load_experiment
from config.runtime import RunContext
from training.checkpointing import CheckpointRef
from training.checkpointing.base import ExperimentFileStore
from training.engine import Engine
from utils import get_root_dir
from utils.logging import setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)
warnings.filterwarnings("ignore")

app = typer.Typer(pretty_exceptions_enable=False)


def _experiment_file_path(name: str) -> Path:
    return (
        get_root_dir(markers=("pyproject.toml",))
        / "src"
        / "config"
        / "experiments"
        / f"{name}.py"
    )


def _model_source_file(model: torch.nn.Module) -> Path:
    return (
        get_root_dir(markers=("pyproject.toml",))
        / "src"
        / (f"{model.__module__}".replace(".", os.sep) + ".py")
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    run_name: str = typer.Option(
        "",
        "--name",
        "-n",
        help="MLflow run name",
    ),
    run_suffix: str = typer.Option(
        "",
        "--suffix",
        "-s",
        help="MLflow run suffix (to model name)",
    ),
    compile: str = typer.Option(
        "",
        "--compile",
        "-c",
        help="Compile mode for torch.compile",
    ),
    fullgraph: bool = typer.Option(
        False,
        "--fullgraph",
        help="Require fullgraph during compilation",
    ),
    load_id: str = typer.Option(
        "",
        "--load-id",
        help="MLflow run id to load checkpoint from",
    ),
    load_epoch: int | None = typer.Option(
        None,
        "--load-epoch",
        help="Epoch to load checkpoint from",
    ),
    parent_id: str | None = typer.Option(
        None,
        "--parent-id",
        help="MLflow run id to set as parent",
    ),
    start_epoch: int | None = typer.Option(
        None,
        "--start-epoch",
        help="Epoch to start training from",
    ),
    subsample: int | None = typer.Option(
        None,
        help="Sub sample of dataset for testing",
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show Epoch/Example progress bars",
    ),
    gpu: bool = typer.Option(
        True,
        "--gpu/--no-gpu",
        help="Use GPU as device",
    ),
    model_only: bool = typer.Option(
        False,
        "--model-only",
        help="Load only model weights on resume (skip optimizer/scheduler)",
    ),
    source_backend: str | None = source_backend_arg(),
    source_opts: list[str] = source_opt_arg(),
    source_base_dir: Path | None = source_base_dir_arg(),
    source_volume: str | None = source_volume_arg(),
    tracking_uri: str | None = typer.Option(
        None,
        "--tracking-uri",
        help="Override MLflow tracking URI",
    ),
    artifact_loc: Path | None = typer.Option(
        None,
        "--artifact-loc",
        help="Override artifact storage location",
    ),
) -> None:
    """Train an experiment using dependency-injected configuration."""
    assert not (run_name and run_suffix), (
        "Either 'run-name' or 'run-suffix' must be specified"
    )
    assert run_name or run_suffix, (
        "One of 'run-name' or 'run-suffix' must be specified"
    )
    assert (load_id and load_epoch is not None) or (
        load_epoch is None and not load_id
    ), "load id/epoch only work together"

    device = torch.device(
        "cuda" if torch.cuda.is_available() and gpu else "cpu"
    )
    assert device.type == "cuda" or not gpu, (
        "No GPU available on this device, use --no-gpu option"
    )

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

    runtime = RunContext(
        device=device,
        dtype=torch.float32,
        compile_mode=compile,
        fullgraph=fullgraph,
        subsample=subsample,
    )

    root_obj = ctx.find_root().obj
    experiment_name: str = root_obj["experiment_name"]
    exp = load_experiment(
        experiment_name, runtime, env=env, source_backend=source_backend_obj
    )

    if runtime.compile_mode:
        exp.model.compile(
            mode=runtime.compile_mode, fullgraph=runtime.fullgraph
        )

    start_epoch_int = (
        start_epoch
        if start_epoch is not None
        else (load_epoch + 1 if load_epoch is not None else 1)
    )

    model_name = type(exp.model).__name__
    if run_suffix:
        run_name = f"{model_name}-{run_suffix}"
    if subsample:
        run_name += "-DRY"

    logger.info(
        f'Run: "{run_name}" (Model: {model_name}) | Device: {device}'
        f'{" | DRY-RUN" if subsample else ""}'
    )

    mlflow.set_tracking_uri(env.tracking_uri)
    mlflow.set_experiment(exp.experiment_name)
    logger.info(f"Mlflow connection established at {env.tracking_uri}")

    if load_id and load_epoch is not None:
        checkpoint = exp.checkpoints.load(
            ref=CheckpointRef(run_id=load_id, epoch=load_epoch),
            map_location=device,
        )
        exp.model.load_state_dict(checkpoint.model)
        logger.info("Model state loaded")
        if not model_only:
            exp.strategy.load_optimizer_state(
                optimizer=checkpoint.optimizer,
                scheduler=checkpoint.scheduler,
            )
            logger.info("Optimizer and scheduler state restored")

    progress_bars = build_progress_bars(disable=not progress)
    for pb in progress_bars:
        try:
            pb.start()
        except RuntimeError:
            # build_epoch_progress is already started.
            pass
    engine = Engine(
        experiment=exp, runtime=runtime, progress=progress_bars
    )

    with mlflow.start_run(run_name=run_name, parent_run_id=parent_id):
        mlf_run = mlflow.active_run()
        assert mlf_run is not None

        experiment_file_path = _experiment_file_path(experiment_name)
        if experiment_file_path.exists() and isinstance(
            exp.checkpoints, ExperimentFileStore
        ):
            exp.checkpoints.save_experiment_file(path=experiment_file_path)

        model_file = _model_source_file(exp.model)
        if model_file.exists():
            mlflow.log_artifact(str(model_file))

        mlflow.log_params(
            {
                "experiment_name": exp.experiment_name,
                "model.class": model_name,
                "train.examples": len(exp.train_loader.dataset),
                "val.examples": len(exp.val_loader.dataset),
            }
        )

        engine.fit(start_epoch=start_epoch_int, run_id=mlf_run.info.run_id)


if __name__ == "__main__":
    app()
