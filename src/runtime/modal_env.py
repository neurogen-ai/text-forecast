"""Helpers for deriving a Modal container Env from a job Env.

Split out of ``runtime.modal_runtime`` so these pure functions can be tested
without importing the optional ``modal`` package.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

from config.env import Env, ModalRuntimeConfig


def remote_tracking_uri(env: Env, cfg: ModalRuntimeConfig) -> str:
    """Return the MLflow tracking URI the container (and client) must use.

    A Modal-specific override wins. Otherwise the job's tracking URI is used,
    but a loopback URI (127.0.0.1 / localhost) is rejected because a Modal
    container cannot reach the developer's machine.
    """
    if cfg.tracking_uri:
        return cfg.tracking_uri
    uri = env.tracking_uri
    host = urlparse(uri).hostname
    if host in ("127.0.0.1", "localhost"):
        raise ValueError(
            "Modal runs need a network-reachable MLflow tracking URI. "
            f"env.tracking_uri resolves to loopback {uri!r}; set "
            "[runtime.modal].tracking_uri in config.toml to a remote "
            "tracking server."
        )
    return uri


def container_env(env: Env, cfg: ModalRuntimeConfig, tracking_uri: str) -> Env:
    """Map the job Env onto the modal volume mounts.

    ``artifact_loc`` becomes the checkpoint-volume mount (scratch/cache for
    ``torch.save`` and the source for ``mlflow.log_artifact``), and the source
    default flips to the Modal backend so experiment ``build`` resolves
    datasets against the staged volume.
    """
    return replace(
        env,
        tracking_uri=tracking_uri,
        artifact_loc=Path(f"/modal/{cfg.checkpoint_volume}"),
        source={
            "default": "modal",
            "modal": {"volume": cfg.staged_volume},
        },
    )
