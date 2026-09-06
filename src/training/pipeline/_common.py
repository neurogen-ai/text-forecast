"""Helpers shared by the train/eval pipelines."""

from __future__ import annotations

import importlib
import logging
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
    dtype: str = "fp32",
) -> RunContext:
    """Build the ``RunContext`` for this venue.

    Device selection is an execution detail: cuda when available and allowed,
    cpu otherwise.  The job never carries a device.

    ``dtype`` selects the compute dtype for model parameters and the big
    retriever tensors: ``fp32`` (default), ``bf16``, or ``fp16``. There is no
    autocast or ``GradScaler`` in the training loop, so raw ``fp16`` will
    underflow gradients in AdamW — prefer ``bf16`` unless you add loss
    scaling.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    assert device.type == "cuda" or not gpu, (
        "No GPU available on this venue, use --no-gpu"
    )
    dtypes = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    if dtype not in dtypes:
        raise ValueError(
            f"Unknown dtype {dtype!r}; expected one of {sorted(dtypes)}"
        )
    if dtype == "fp16":
        logging.getLogger(__name__).warning(
            "fp16 requested but the training loop has no GradScaler; "
            "gradients may underflow. Consider bf16 instead."
        )
    return RunContext(
        device=device,
        dtype=dtypes[dtype],
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
