"""Tests for Modal container-Env derivation (2.0 step 6).

These cover the pure helpers in ``runtime.modal_env`` so they can run without
the optional ``modal`` package installed.
"""

from __future__ import annotations

from pathlib import Path

from config.env import Env, ModalRuntimeConfig
from runtime.modal_env import container_env, remote_tracking_uri

import pytest


def _env(tracking_uri: str = "http://127.0.0.1:5000") -> Env:
    return Env(
        tracking_uri=tracking_uri,
        artifact_loc=Path("/tmp/artifacts"),
        runtime={},
        source={
            "default": "local",
            "local": {"base_dir": "/tmp/data"},
        },
    )


def _cfg(**overrides) -> ModalRuntimeConfig:
    base = dict(
        project="citef",
        staged_volume="openalex-staged",
        checkpoint_volume="cf-checkpoints",
    )
    base.update(overrides)
    return ModalRuntimeConfig(**base)


def test_remote_tracking_uri_uses_override_when_set() -> None:
    cfg = _cfg(tracking_uri="https://mlflow.example.com")
    assert (
        remote_tracking_uri(_env(), cfg) == "https://mlflow.example.com"
    )


def test_remote_tracking_uri_inherits_env_without_override() -> None:
    env = _env(tracking_uri="http://mlflow:5000")
    assert remote_tracking_uri(env, _cfg()) == "http://mlflow:5000"


def test_remote_tracking_uri_rejects_loopback_host() -> None:
    for uri in ("http://127.0.0.1:5000", "http://localhost:5000"):
        with pytest.raises(ValueError, match="network-reachable"):
            remote_tracking_uri(_env(tracking_uri=uri), _cfg())


def test_remote_tracking_uri_override_wins_over_loopback() -> None:
    cfg = _cfg(tracking_uri="https://mlflow.example.com")
    assert remote_tracking_uri(_env(), cfg) == "https://mlflow.example.com"


def test_container_env_maps_artifact_loc_to_checkpoint_volume() -> None:
    env = container_env(
        _env(tracking_uri="http://mlflow:5000"),
        _cfg(),
        tracking_uri="http://mlflow:5000",
    )
    assert env.artifact_loc.as_posix() == "/modal/cf-checkpoints"
    assert env.tracking_uri == "http://mlflow:5000"


def test_container_env_switches_source_backend_to_modal() -> None:
    env = container_env(_env(), _cfg(), tracking_uri="http://mlflow:5000")
    assert env.source["default"] == "modal"
    assert env.source["modal"]["volume"] == "openalex-staged"


def test_container_env_respects_custom_volume_labels() -> None:
    cfg = _cfg(staged_volume="my-stage", checkpoint_volume="my-cp")
    env = container_env(_env(), cfg, tracking_uri="http://mlflow:5000")
    assert env.artifact_loc.as_posix() == "/modal/my-cp"
    assert env.source["modal"]["volume"] == "my-stage"
