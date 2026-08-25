"""Tests for 2.0 step 4: runtime dispatch plumbing."""

import pytest

from config.loader import read_experiment_name
from runtime.local import LocalRuntime
from training.pipeline.eval import EvalJob
from training.pipeline.train import TrainJob


def test_read_experiment_name_real_module():
    assert (
        read_experiment_name("transformer_class")
        == "Basic-TransformerClass-cited-gr-1"
    )


def test_read_experiment_name_unknown_module():
    with pytest.raises(ValueError, match="Unknown experiment"):
        read_experiment_name("does-not-exist")


def test_read_experiment_name_missing_declaration(tmp_path, monkeypatch):
    import config.loader as loader

    fake_dir = tmp_path / "experiments"
    fake_dir.mkdir()
    (fake_dir / "bare.py").write_text("x = 'no declaration here'\n")
    monkeypatch.setattr(loader, "_EXPERIMENTS_DIR", fake_dir)
    with pytest.raises(ValueError, match="experiment_name"):
        loader.read_experiment_name("bare")


def test_runtimes_implement_train_eval_dispatch():
    for method in ("run_train", "run_eval"):
        assert callable(getattr(LocalRuntime, method))


def test_job_progress_and_suffix_fields():
    job_fields = TrainJob.__dataclass_fields__
    assert "progress" in job_fields and "run_suffix" in job_fields
    assert "module_name" in EvalJob.__dataclass_fields__
    assert "progress" in EvalJob.__dataclass_fields__
