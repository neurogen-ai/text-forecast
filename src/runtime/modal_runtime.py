"""Modal runtime: volumes, image, GPU embedder, and remote job dispatch."""

from dataclasses import replace
from logging import getLogger
from pathlib import Path
from typing import Any

import modal

from config.env import Env, load_env, modal_runtime_config
from config.loader import read_experiment_name
from data.pipeline.describe import DescribeJob, run_describe_pipeline
from data.pipeline.engineer import EngineerJob, run_engineer_pipeline
from data.preprocess.embed_modal import ModalEmbedder
from data.preprocess.embed_huggingface import EMBEDDERS, EmbedderConfig
from data.preprocess.pipeline import PreprocessJob, run_preprocess_pipeline
from data.sources.base import DataSource
from runtime.base import RunResult, Runtime, TextEmbedder
from runtime.modal_env import container_env, remote_tracking_uri
from runtime.run_lifecycle import create_run
from training.pipeline.eval import EvalJob, run_eval_pipeline
from training.pipeline.train import TrainJob, run_train_pipeline
from utils.logging import setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)

# Volume labels mounted by the remote preprocess function and GPU embedder. These
# are module constants because Modal function decorators are evaluated at import time.
VOLUME_LABEL = "openalex-staged"
EMBEDDERS_VOLUME_LABEL = "embedders"
EMBEDDERS_MOUNT = "/embedders"

# Load runtime config at module import time so Modal image/function decorators can
# respond to config.toml. Local Python does not need to match the Modal image Python.
_ENV = load_env()
_MODAL_CONFIG = _ENV.runtime.get("modal", {})
_GPU_TYPE = _MODAL_CONFIG.get("gpu", "L4")
_PYTHON_VERSION = _MODAL_CONFIG.get("python_version", "3.12")
_EMBEDDER_BATCH_SIZE = int(_MODAL_CONFIG.get("embedder_batch_size", 64))

# Training/eval volume and GPU settings (2.0 step 6).
#
# ``checkpoint_volume`` falls back to a default rather than raising here:
# Modal resolves this class by module reference, so the *container* re-imports
# this module with whatever config.toml was baked into the image. A missing
# key must not crash container import. Client-side dispatch still fails fast
# via ``modal_runtime_config(require_checkpoint_volume=True)``.
_CHECKPOINT_VOLUME_LABEL = _MODAL_CONFIG.get("checkpoint_volume", "cf-checkpoints")
_TRAIN_GPU = _MODAL_CONFIG.get("train_gpu", "A10G")
_TRAIN_TIMEOUT = int(_MODAL_CONFIG.get("timeout", 86400))
STAGED_VOLUME_LABEL = _MODAL_CONFIG.get("staged_volume", "openalex-staged")

# GPU snapshotting gives fast cold starts for the training container, but
# ``torch.compile`` cannot survive a snapshot restore, so the dispatcher
# clears ``compile_mode`` on the job whenever this is enabled (plan 2.0
# §10.1 caveat).
_SNAPSHOT_ENABLED = True

# Training shares the preprocessing image: pyproject deps + flat src-layout.
# config/ is baked in too, deviating from plan 2.0 §10.3, because this module
# calls load_env() at import time (client-side for the decorators below, and
# again inside the container when Modal re-imports it by module reference).
# Jobs still carry their own Env; the baked config only feeds these constants.

logger.debug(
    "Modal runtime config: gpu=%r python_version=%r embedder_batch_size=%r "
    "checkpoint_volume=%r train_gpu=%r timeout=%r",
    _GPU_TYPE,
    _PYTHON_VERSION,
    _EMBEDDER_BATCH_SIZE,
    _CHECKPOINT_VOLUME_LABEL,
    _TRAIN_GPU,
    _TRAIN_TIMEOUT,
)

preprocess_image = (
    modal.Image.debian_slim(python_version=_PYTHON_VERSION)
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .add_local_dir("config", remote_path="/root/config", copy=True)
    .env({"PYTHONPATH": "/root/src"})
)

# Shared by the preprocess, describe, and engineer remote functions.
MODAL_APP_NAME = "text-forecast-data"

app = modal.App(MODAL_APP_NAME)
volume = modal.Volume.from_name(VOLUME_LABEL, create_if_missing=True)
embedders_volume = modal.Volume.from_name(
    EMBEDDERS_VOLUME_LABEL, create_if_missing=True
)
train_staged_volume = modal.Volume.from_name(
    STAGED_VOLUME_LABEL, create_if_missing=True
)
train_checkpoint_volume = modal.Volume.from_name(
    _CHECKPOINT_VOLUME_LABEL, create_if_missing=True
)


@app.cls(
    gpu=_GPU_TYPE,
    max_containers=1,
    timeout=600,
    scaledown_window=300,
    image=preprocess_image,
    volumes={EMBEDDERS_MOUNT: embedders_volume},
    #enable_memory_snapshot=True,
    #experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class ModalEmbeddingGPU:
    """GPU embedder backed by a HuggingFace sentence encoder.

    Models are cached in the ``embedders`` volume before falling back to a
    HuggingFace download. The container state is snapshotted after model load
    for fast cold starts. ``torch.compile`` is intentionally disabled because
    compilation breaks snapshot restore.
    """

    model_name: str = modal.parameter(default="answerdotai/ModernBERT-base")

    @modal.enter()
    def setup(self) -> None:
        logger.info(
            "ModalEmbeddingGPU.setup: loading embedder model_name=%r on gpu=%r",
            self.model_name,
            _GPU_TYPE,
        )
        from data.preprocess.embed_huggingface import HuggingFaceEmbedder

        logger.info("ModalEmbeddingGPU.setup: creating HuggingFaceEmbedder")
        self._embedder = HuggingFaceEmbedder(
            model_name=self.model_name,
            device="cuda",
            dtype="bfloat16",
            compile=True,
            cache_dir=EMBEDDERS_MOUNT,
        )
        logger.debug("ModalEmbeddingGPU.setup: calling embedder.load()")
        self._embedder.load()
        logger.info("ModalEmbeddingGPU.setup: embedder loaded successfully")

    @modal.method()
    def encode(self, texts: list[str]) -> list[list[float]]:
        logger.debug(
            "ModalEmbeddingGPU.encode: encoding batch of %d texts with model_name=%r",
            len(texts),
            self.model_name,
        )
        result = self._embedder.encode(texts)
        logger.debug(
            "ModalEmbeddingGPU.encode: finished batch of %d texts",
            len(texts),
        )
        return result



_train_cls_kwargs: dict[str, Any] = dict(
    gpu=_TRAIN_GPU,
    image=preprocess_image,
    volumes={
        f"/modal/{STAGED_VOLUME_LABEL}": train_staged_volume,
        f"/modal/{_CHECKPOINT_VOLUME_LABEL}": train_checkpoint_volume,
    },
    timeout=_TRAIN_TIMEOUT,
    scaledown_window=300,
)
if _SNAPSHOT_ENABLED:
    _train_cls_kwargs["enable_memory_snapshot"] = True
    _train_cls_kwargs["experimental_options"] = {"enable_gpu_snapshot": True}


@app.cls(**_train_cls_kwargs)
class ModalTrainingGPU:
    """Remote training/eval GPU container (2.0 step 6).

    Runs the venue-independent train/eval pipelines headless (``progress=None``,
    plan §11). The image and GPU come from config; the staged and checkpoint
    volumes are mounted exactly where ``container_env`` maps paths.

    ``torch.compile`` is disabled by dropping ``compile_mode`` on the job at
    dispatch time when snapshotting is enabled, because compilation breaks
    snapshot restore. Experiment files never adapt to this (plan P1).
    """

    @modal.enter(snap=True)
    def setup(self) -> None:
        logger.info(
            "ModalTrainingGPU.setup: container ready gpu=%r", _TRAIN_GPU
        )

    @modal.method()
    def train(self, job: TrainJob) -> RunResult:
        logger.info("ModalTrainingGPU.train: starting train job")
        return run_train_pipeline(job)

    @modal.method()
    def eval(self, job: EvalJob) -> RunResult:
        logger.info("ModalTrainingGPU.eval: starting eval job")
        return run_eval_pipeline(job)


class _InContainerRuntime:
    """Runtime used inside Modal containers.

    Dataframe work stays in the CPU container while embedding calls are
    delegated to the ``ModalEmbeddingGPU`` class defined in this module.
    """

    supported_source_backends = {"modal"}

    def get_embedder(self, key: str, **kwargs: Any) -> TextEmbedder:
        logger.debug("_InContainerRuntime.get_embedder: key=%r kwargs=%r", key, kwargs)
        return _build_modal_embedder(key, **kwargs)

    def run_preprocess(self, job: PreprocessJob) -> DataSource:
        logger.info("_InContainerRuntime.run_preprocess: starting job")
        return run_preprocess_pipeline(job, self)


def _in_container_runtime() -> _InContainerRuntime:
    """Return the runtime used inside Modal containers.

    Volume-backed sources resolve against the in-container mount roots and
    embedders route to the GPU class in this module. Shared by every remote
    entry point so container-side behaviour stays identical across jobs.
    """
    return _InContainerRuntime()


@app.function(
    volumes={f"/modal/{VOLUME_LABEL}": volume},
    image=preprocess_image,
    max_containers=1,
    timeout=3600,
)
def run_preprocess_remote(job: PreprocessJob) -> DataSource:
    """Container entry point for the full preprocessing pipeline.

    Volumes are mounted at the same paths ``ModalDataSource.resolve()``
    returns. The pipeline runs in this CPU container, but embedders are routed
    through ``ModalEmbedder`` so batches execute on ``ModalEmbeddingGPU``.
    """
    logger.info(
        "run_preprocess_remote: starting preprocess job origin=%r destination=%r",
        getattr(job.origin, "name", job.origin),
        getattr(job.destination, "name", job.destination),
    )
    logger.debug(
        "run_preprocess_remote: embedder_key=%r embed_cols=%r",
        job.embedder_key,
        job.embed_cols,
    )
    result = run_preprocess_pipeline(job, _in_container_runtime())
    logger.info("run_preprocess_remote: preprocess job completed")
    return result


@app.function(
    volumes={f"/modal/{VOLUME_LABEL}": volume},
    image=preprocess_image,
    max_containers=1,
    timeout=3600,
)
def run_describe_remote(job: DescribeJob) -> None:
    """Container entry point for the describe pipeline.

    The staged volume is mounted at the same path ``ModalDataSource.resolve()``
    returns, so the job's ``DataSource`` resolves inside the container.
    """
    logger.info(
        "run_describe_remote: starting describe job source=%r",
        getattr(job.source, "name", job.source),
    )
    run_describe_pipeline(job)
    logger.info("run_describe_remote: describe job completed")


@app.function(
    volumes={f"/modal/{VOLUME_LABEL}": volume},
    image=preprocess_image,
    max_containers=1,
    timeout=3600,
)
def run_engineer_remote(job: EngineerJob) -> DataSource:
    """Container entry point for the engineer pipeline."""
    logger.info(
        "run_engineer_remote: starting engineer job origin=%r output=%r",
        getattr(job.origin, "name", job.origin),
        getattr(job.output, "name", job.output),
    )
    result = run_engineer_pipeline(job)
    logger.info("run_engineer_remote: engineer job completed")
    return result


class ModalRuntime:
    """Modal backend: volume sources, GPU embedder, and remote job dispatch."""

    supported_source_backends = {"modal"}

    def __init__(self, env: Env, project: str) -> None:
        self._env = env
        self._project = project

        configured_volume = env.source.get("modal", {}).get("volume")
        if configured_volume != VOLUME_LABEL:
            raise ValueError(
                "Modal volume label in config/config.toml must match the "
                f"module constant: {VOLUME_LABEL!r}. Got: {configured_volume!r}."
            )

    def get_embedder(self, key: str, **kwargs: Any) -> TextEmbedder:
        logger.debug("ModalRuntime.get_embedder: key=%r kwargs=%r", key, kwargs)
        return _build_modal_embedder(key, **kwargs)

    def get_source(self, root: str | Path, name: str) -> DataSource:
        # ``root`` is a volume mount root such as "/modal/<volume-label>";
        # its last segment is the volume label.
        return ModalDataSource(volume=Path(root).name, name=name)

    def run_preprocess(self, job: PreprocessJob) -> DataSource:
        logger.info(
            "ModalRuntime.run_preprocess: dispatching job to Modal project %r",
            self._project,
        )
        with modal.enable_output():
            fn = modal.Function.from_name(MODAL_APP_NAME, "run_preprocess_remote")
            result = fn.remote(job)
        logger.info("ModalRuntime.run_preprocess: received result from Modal")
        return result

    def run_describe(self, job: DescribeJob) -> None:
        logger.info(
            "ModalRuntime.run_describe: dispatching job to Modal project %r",
            self._project,
        )
        with modal.enable_output():
            fn = modal.Function.from_name(MODAL_APP_NAME, "run_describe_remote")
            fn.remote(job)
        logger.info("ModalRuntime.run_describe: received result from Modal")

    def run_engineer(self, job: EngineerJob) -> DataSource:
        logger.info(
            "ModalRuntime.run_engineer: dispatching job to Modal project %r",
            self._project,
        )
        with modal.enable_output():
            fn = modal.Function.from_name(MODAL_APP_NAME, "run_engineer_remote")
            result = fn.remote(job)
        logger.info("ModalRuntime.run_engineer: received result from Modal")
        return result

    def run_train(self, job: TrainJob) -> RunResult:
        """Create the run client-side, then spawn a remote training container.

        Fire-and-forget dispatch (plan P7): the CLI returns immediately with
        the MLflow run id and the Modal FunctionCall id so the caller can keep
        following the job via ``modal.FunctionCall.from_id``.
        """
        cfg = modal_runtime_config(self._env, require_checkpoint_volume=True)
        tracking_uri = remote_tracking_uri(job.env, cfg)

        experiment_name = read_experiment_name(job.experiment_name)
        provisional_name = job.run_name or job.run_suffix or "train"
        run_id = create_run(
            env=replace(job.env, tracking_uri=tracking_uri),
            experiment_name=experiment_name,
            run_name=provisional_name,
            parent_id=job.parent_id,
        )

        dispatch_compile = "" if _SNAPSHOT_ENABLED else job.compile_mode
        if _SNAPSHOT_ENABLED and job.compile_mode:
            logger.warning(
                "torch.compile cleared for Modal snapshot restore: "
                "compile_mode=%r dropped", job.compile_mode
            )
        container_job = replace(
            job,
            env=container_env(job.env, cfg, tracking_uri),
            run_id=run_id,
            compile_mode=dispatch_compile,
        )

        logger.info(
            "ModalRuntime.run_train: spawning train run=%r on %r",
            run_id, _TRAIN_GPU,
        )
        with modal.enable_output():
            call = ModalTrainingGPU().train.spawn(container_job)
        logger.info(
            "ModalRuntime.run_train: spawned run=%r function_call_id=%r",
            run_id, call.object_id,
        )
        return RunResult(
            run_id=run_id,
            status="spawned",
            checkpoints=[],
            metrics={},
            modal_function_call_id=str(call.object_id),
        )

    def run_eval(self, job: EvalJob) -> RunResult:
        """Create the eval run client-side, then spawn a remote eval container."""
        cfg = modal_runtime_config(self._env, require_checkpoint_volume=True)
        tracking_uri = remote_tracking_uri(job.env, cfg)

        training_experiment = (
            read_experiment_name(job.module_name) if job.module_name else None
        )
        eval_experiment = job.experiment_name or (
            f"{training_experiment}-EVAL"
            if training_experiment is not None
            else f"eval-{job.run_id}"
        )
        eval_run_id = create_run(
            env=replace(job.env, tracking_uri=tracking_uri),
            experiment_name=eval_experiment,
            run_name=f"{job.prefix}eval-{job.run_id}",
        )

        container_job = replace(
            job,
            env=container_env(job.env, cfg, tracking_uri),
            eval_run_id=eval_run_id,
        )

        logger.info(
            "ModalRuntime.run_eval: spawning eval run=%r on %r",
            eval_run_id, _TRAIN_GPU,
        )
        with modal.enable_output():
            call = ModalTrainingGPU().eval.spawn(container_job)
        logger.info(
            "ModalRuntime.run_eval: spawned eval_run=%r function_call_id=%r",
            eval_run_id, call.object_id,
        )
        return RunResult(
            run_id=eval_run_id,
            status="spawned",
            metrics={},
            modal_function_call_id=str(call.object_id),
        )


def _embedder_config(key: str) -> EmbedderConfig:
    """Return the configuration for a local embedder registry key."""
    try:
        return EMBEDDERS[key]
    except KeyError:
        raise ValueError(
            f"Modal runtime does not support embedder {key!r}. "
            f"Supported keys: {list(EMBEDDERS)}"
        ) from None


def _key_to_model_name(key: str) -> str:
    return _embedder_config(key).model_name


def _resolve_output_dim(model_name: str) -> int:
    """Return the embedding dimension for a known HuggingFace model name."""
    for config in EMBEDDERS.values():
        if config.model_name == model_name:
            return config.output_dim

    raise ValueError(
        f"Unknown output dimension for {model_name!r}. "
        "Add it to EMBEDDERS in src/runtime/modal_runtime.py."
    )

def _build_modal_embedder(key: str, **kwargs: Any) -> TextEmbedder:
    logger.debug("_build_modal_embedder: key=%r kwargs=%r", key, kwargs)
    model_name = kwargs.get("model_name") or _key_to_model_name(key)
    output_dim = _resolve_output_dim(model_name)
    batch_size = int(
        kwargs.get("batch_size")
        or _EMBEDDER_BATCH_SIZE
    )
    logger.debug(
        "_build_modal_embedder: model_name=%r output_dim=%d batch_size=%d",
        model_name,
        output_dim,
        batch_size,
    )
    return ModalEmbedder(
        model_name=model_name,
        output_dim=output_dim,
        batch_size=batch_size,
        gpu_cls=ModalEmbeddingGPU,
    )
