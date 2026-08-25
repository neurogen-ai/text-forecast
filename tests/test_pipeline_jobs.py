"""Tests for the train/eval pipeline jobs (2.0 step 3)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from config.env import Env
from runtime.base import RunResult
from training.pipeline import EvalJob, TrainJob, run_eval_pipeline
from training.pipeline._common import (
    build_run_context,
    experiment_file,
    model_source_file,
)


def _env() -> Env:
    return Env(
        tracking_uri="http://127.0.0.1:5000",
        artifact_loc="/tmp/artifacts",
    )


def _train_job() -> TrainJob:
    return TrainJob(
        experiment_name="graph_embed_class",
        experiment_source=b"# experiment module",
        run_name="test-run",
        env=_env(),
    )


def _eval_job() -> EvalJob:
    return EvalJob(
        run_id="abc123",
        epoch=1,
        start_date=datetime(1990, 1, 1),
        interval=1,
        env=_env(),
    )


class TestTrainJob:
    def test_frozen(self) -> None:
        job = _train_job()
        with pytest.raises(AttributeError):
            job.run_name = "mutated"  # type: ignore[misc]

    def test_replace_stamps_run_id(self) -> None:
        job = replace(_train_job(), run_id="run-42")
        assert job.run_id == "run-42"
        assert job.experiment_name == "graph_embed_class"


class TestEvalJob:
    def test_frozen(self) -> None:
        job = _eval_job()
        with pytest.raises(AttributeError):
            job.epoch = 2  # type: ignore[misc]

    def test_defaults(self) -> None:
        job = _eval_job()
        assert job.interval_unit == "y"
        assert job.prefix == ""
        assert job.eval_run_id == ""
        assert job.clean_up is False


class TestRunResult:
    def test_defaults(self) -> None:
        result = RunResult(run_id="r", status="completed")
        assert result.checkpoints == []
        assert result.metrics == {}
        assert result.modal_function_call_id is None


class TestHelpers:
    def test_experiment_file_roundtrip_and_cleanup(self) -> None:
        with experiment_file(b"x = 1") as path:
            assert path.read_bytes() == b"x = 1"
            parent = path.parent
        assert not parent.exists()

    def test_build_run_context_cpu(self) -> None:
        ctx = build_run_context(gpu=False, compile_mode="", subsample=8)
        assert ctx.device.type == "cpu"
        assert ctx.subsample == 8


class TestEvalFailFast:
    def test_missing_eval_run_id_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "training.checkpointing.mlflow_store.MlflowCheckpointProcessor."
            "__post_init__",
            lambda self: None,
        )
        job = _eval_job()
        with pytest.raises(ValueError, match="eval_run_id is empty"):
            run_eval_pipeline(job)

    def test_no_experiment_file_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "training.checkpointing.mlflow_store.MlflowCheckpointProcessor."
            "__post_init__",
            lambda self: None,
        )
        monkeypatch.setattr(
            "training.checkpointing.mlflow_store.MlflowCheckpointProcessor."
            "experiment_file_exists",
            lambda self, *, run_id: False,
        )
        job = replace(_eval_job(), eval_run_id="eval-run-1")
        with pytest.raises(ValueError, match="No experiment file found"):
            run_eval_pipeline(job)


def test_model_source_file_resolves_engine() -> None:
    from training.engine import Engine

    file = model_source_file(Engine)
    assert file is not None
    assert file.name == "engine.py"
