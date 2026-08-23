"""Modal runtime: volumes, image, GPU embedder, and remote job dispatch."""

from logging import getLogger
from pathlib import Path
from typing import Any

import modal

from config.env import load_env
from data.pipeline.describe import DescribeJob
from data.pipeline.engineer import EngineerJob
from data.preprocess.embed_modal import ModalEmbedder
from data.preprocess.embed_huggingface import EMBEDDERS, EmbedderConfig
from data.preprocess.pipeline import PreprocessJob, run_preprocess_pipeline
from data.sources.base import DataSource
from utils.logging import setup_logger

from config.env import Env
from runtime.base import Runtime, TextEmbedder

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

logger.debug(
    "Modal runtime config: gpu=%r python_version=%r embedder_batch_size=%r",
    _GPU_TYPE,
    _PYTHON_VERSION,
    _EMBEDDER_BATCH_SIZE,
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
MODAL_APP_NAME = "citef-data"

app = modal.App(MODAL_APP_NAME)
volume = modal.Volume.from_name(VOLUME_LABEL, create_if_missing=True)
embedders_volume = modal.Volume.from_name(
    EMBEDDERS_VOLUME_LABEL, create_if_missing=True
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
        raise NotImplementedError(
            "Modal dispatch for describe is added in plan 1.3 step 5."
        )

    def run_engineer(self, job: EngineerJob) -> DataSource:
        raise NotImplementedError(
            "Modal dispatch for engineer is added in plan 1.3 step 5."
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
