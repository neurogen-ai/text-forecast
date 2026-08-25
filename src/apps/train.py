from __future__ import annotations

import warnings
from logging import getLogger
from pathlib import Path

import typer

from config.env import load_env
from runtime.factory import build_runtime
from training.pipeline.train import TrainJob
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
    runtime_name: str | None = typer.Option(
        None,
        "--runtime",
        help="Execution backend (local or modal); overrides [runtime].default",
    ),
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
    experiment_name: str = root_obj["experiment_name"]

    job = TrainJob(
        experiment_name=experiment_name,
        experiment_source=_experiment_file_path(experiment_name).read_bytes(),
        run_name=run_name,
        run_suffix=run_suffix,
        parent_id=parent_id,
        start_epoch=start_epoch,
        compile_mode=compile,
        fullgraph=fullgraph,
        load_id=load_id,
        load_epoch=load_epoch,
        model_only=model_only,
        subsample=subsample,
        gpu=gpu,
        progress=progress,
        env=env,
    )

    result = runtime.run_train(job)
    logger.info(
        f"Run {result.run_id} {result.status}"
        f" | checkpoints: {len(result.checkpoints)}"
    )


if __name__ == "__main__":
    app()
