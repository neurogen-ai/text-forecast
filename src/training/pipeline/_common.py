"""Helpers shared by the train/eval pipelines."""

from __future__ import annotations

import importlib
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

from config.runtime import RunContext


@contextmanager
def experiment_file(source: bytes) -> Iterator[Path]:
    """Write raw experiment-file bytes to a temp file and clean up after.

    Both venues use this: the client reads its local experiment module file,
    the container receives the bytes over the job.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="cf-experiment-"))
    path = tmp_dir / "experiment.py"
    path.write_bytes(source)
    try:
        yield path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def build_run_context(
    *,
    gpu: bool = True,
    compile_mode: str = "",
    fullgraph: bool = False,
    subsample: int | None = None,
) -> RunContext:
    """Build the ``RunContext`` for this venue.

    Device selection is an execution detail: cuda when available and allowed,
    cpu otherwise.  The job never carries a device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    assert device.type == "cuda" or not gpu, (
        "No GPU available on this venue, use --no-gpu"
    )
    return RunContext(
        device=device,
        dtype=torch.float32,
        compile_mode=compile_mode,
        fullgraph=fullgraph,
        subsample=subsample,
    )


def model_source_file(model: torch.nn.Module | type) -> Path | None:
    """Locate the source file defining a model's class.

    Accepts an instance or a class.  Resolved through the imported module
    rather than a project-root marker so it works both locally and inside a
    container with ``src`` on PYTHONPATH.
    """
    cls = model if isinstance(model, type) else type(model)
    try:
        module = importlib.import_module(cls.__module__)
    except ImportError:
        return None
    file = getattr(module, "__file__", None)
    return Path(file) if file is not None else None
