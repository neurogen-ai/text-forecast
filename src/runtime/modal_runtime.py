"""Modal runtime: volumes, image, GPU embedder, and remote preprocess dispatch."""


from logging import getLogger
from typing import Any

import modal

from config.env import load_env
from data.preprocess.embed_modal import ModalEmbedder
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

MODAL_APP_NAME = "citef-preprocess"

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


class _ModalRemoteRuntime:
    """Runtime used inside the Modal preprocess container.

    Dataframe work stays in the CPU preprocess container while embedding calls
    are delegated to the ``ModalEmbeddingGPU`` class.
    """

    supported_source_backends = {"modal"}

    def get_embedder(self, key: str, **kwargs: Any) -> TextEmbedder:
        logger.debug("_ModalRemoteRuntime.get_embedder: key=%r kwargs=%r", key, kwargs)
        return _build_modal_embedder(key, **kwargs)

    def run_preprocess(self, job: PreprocessJob) -> DataSource:
        logger.info("_ModalRemoteRuntime.run_preprocess: starting job")
        return run_preprocess_pipeline(job, self)


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
    result = run_preprocess_pipeline(job, _ModalRemoteRuntime())
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


def _key_to_model_name(key: str) -> str:
    """Map a local embedder registry key to its HuggingFace model name."""
    mapping = {
        "modernbert-base": "answerdotai/ModernBERT-base",
        "modernbert-embed-base": "nomic-ai/modernbert-embed-base",
    }

    if key not in mapping:
        raise ValueError(
            f"Modal runtime does not support embedder {key!r}. "
            f"Supported keys: {list(mapping)}"
        )
    return mapping[key]


def _resolve_output_dim(model_name: str) -> int:
    """Return the embedding dimension for known models without loading them."""
    dim_map = {
        "answerdotai/ModernBERT-base": 768,
        "modernbert-base": 768,
        "nomic-ai/modernbert-embed-base": 768,
    }
    dim = dim_map.get(model_name, None)
    if dim is not None:
        return dim
    raise ValueError(
        f"Unknown output dimension for {model_name!r}. "
        "Add it to _resolve_output_dim in src/runtime/modal_runtime.py."
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
