"""Runtime factory: builds the requested local or Modal runtime backend."""

from __future__ import annotations

from config.env import Env

from .base import Runtime


def build_runtime(runtime_name: str, env: Env) -> Runtime:
    """Return a configured runtime backend.

    The ``modal`` module is imported lazily so that local installs without the
    optional ``modal`` extra continue to work.
    """
    if runtime_name == "local":
        from .local import LocalRuntime

        return LocalRuntime()

    if runtime_name == "modal":
        try:
            from .modal_runtime import ModalRuntime
        except ImportError as exc:
            raise RuntimeError(
                "The modal runtime requires the 'modal' extra: "
                "pip install citation-forecast[modal]"
            ) from exc

        modal_config = env.runtime.get("modal", {})
        return ModalRuntime(
            env=env,
            project=modal_config.get("project", "citef"),
        )

    raise ValueError(
        f"Runtime {runtime_name!r} is not supported. "
        "Choose from: local, modal."
    )
