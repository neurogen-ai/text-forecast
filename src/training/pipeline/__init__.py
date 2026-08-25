"""Runtime-agnostic train/eval job specs.

Mirrors the ``src/data/pipeline/`` convention: frozen dataclasses describe a
job and plain functions execute it.  The functions are identical on every
execution venue (local in-process, Modal container); machine-level settings
arrive only through the injected :class:`Env` on the job, and dispatch is
owned by the ``Runtime`` implementations.
"""

from training.pipeline.eval import EvalJob, run_eval_pipeline
from training.pipeline.train import TrainJob, run_train_pipeline

__all__ = [
    "EvalJob",
    "TrainJob",
    "run_eval_pipeline",
    "run_train_pipeline",
]
