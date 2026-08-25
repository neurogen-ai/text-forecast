"""Tests for the typed ``[runtime.modal]`` accessor (2.0 step 5)."""

from __future__ import annotations

import pytest

from config.env import Env, Path, modal_runtime_config


def _env(modal: dict) -> Env:
    return Env(
        tracking_uri="http://127.0.0.1:5000",
        artifact_loc=Path("/tmp/artifacts"),
        runtime={"default": "local", "modal": modal},
    )


def test_defaults_when_section_missing() -> None:
    cfg = modal_runtime_config(_env_modal_absent())
    assert cfg.project == "citef"
    assert cfg.gpu == "L4"
    assert cfg.train_gpu == "A10G"
    assert cfg.checkpoint_volume is None
    assert cfg.timeout == 86400
    assert cfg.tracking_uri is None


def _env_modal_absent() -> Env:
    return Env(
        tracking_uri="http://127.0.0.1:5000",
        artifact_loc=Path("/tmp/artifacts"),
        runtime={"default": "local"},
    )


def test_values_read_from_config() -> None:
    env = _env(
        {
            "project": "myproj",
            "train_gpu": "L4",
            "checkpoint_volume": "ckpts",
            "timeout": 3600,
            "tracking_uri": "https://mlflow.example.com",
        }
    )
    cfg = modal_runtime_config(env)
    assert cfg.project == "myproj"
    assert cfg.train_gpu == "L4"
    assert cfg.checkpoint_volume == "ckpts"
    assert cfg.timeout == 3600
    assert cfg.tracking_uri == "https://mlflow.example.com"


def test_missing_checkpoint_volume_allowed_by_default() -> None:
    cfg = modal_runtime_config(_env({}))
    assert cfg.checkpoint_volume is None


def test_missing_checkpoint_volume_raises_when_required() -> None:
    with pytest.raises(ValueError, match="checkpoint_volume"):
        modal_runtime_config(_env({}), require_checkpoint_volume=True)


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown keys"):
        modal_runtime_config(_env({"gpus": "8"}))
