"""Modal runtime: volumes, image, GPU embedder, and remote preprocess dispatch."""

from __future__ import annotations

from logging import getLogger
from typing import Any, TYPE_CHECKING

import modal

from config.env import Env
from data.preprocess.embed_modal import ModalEmbedder
from data.preprocess.pipeline import PreprocessJob, run_preprocess_pipeline
from data.sources.base import DataSource

if TYPE_CHECKING:
    from runtime.base import Runtime, TextEmbedder

logger = getLogger(__name__)

# Volume label mounted by the remote preprocess function. This is a module
# constant because Modal function decorators are evaluated at import time.
VOLUME_LABEL = "openalex-staged"

preprocess_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .env({"PYTHONPATH": "/root/src"})
)

app = modal.App("citef-preprocess")
volume = modal.Volume.from_name(VOLUME_LABEL, create_if_missing=True)


@app.cls(
    gpu="L4",
    image=preprocess_image,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class ModalEmbeddingGPU:
    """Batched GPU embedder backed by a HuggingFace sentence encoder.

    The container state is snapshotted after model load for fast cold starts.
    ``torch.compile`` is intentionally disabled because compilation breaks
    snapshot restore.
    """

    def __init__(self, model_name: str = "answerdotai/ModernBERT-base") -> None:
        self.model_name = model_name

    @modal.enter(snap=True)
    def setup(self) -> None:
        from data.preprocess.embed_huggingface import HuggingFaceEmbedder

        self._embedder = HuggingFaceEmbedder(
            model_name=self.model_name,
            device="cuda",
            dtype="bfloat16",
            compile=False,
        )

    @modal.batched(max_batch_size=64, wait_ms=200)
    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts)


@app.function(
    volumes={f"/modal/{VOLUME_LABEL}": volume},
    image=preprocess_image,
)
def run_preprocess_remote(job: PreprocessJob) -> DataSource:
    """Container entry point for the full preprocessing pipeline.

    Volumes are mounted at the same paths ``ModalDataSource.resolve()``
    returns, so a lightweight ``LocalRuntime`` can run the pipeline unchanged.
    """
    from runtime.local import LocalRuntime

    local_runtime = LocalRuntime()
    return run_preprocess_pipeline(job, local_runtime)


class ModalRuntime:
    """Modal backend: volume sources, GPU embedder, and remote job dispatch."""

    supported_source_backends = {"modal"}

    def __init__(
        self,
        env: Env,
        project: str,
        gpu: str,
        embedder_batch_size: int,
    ) -> None:
        self._env = env
        self._project = project
        self._gpu_type = gpu
        self._embedder_batch_size = embedder_batch_size

        configured_volume = env.source.get("modal", {}).get("volume")
        if configured_volume != VOLUME_LABEL:
            raise ValueError(
                "Modal volume label in config/config.toml must match the "
                f"module constant: {VOLUME_LABEL!r}. Got: {configured_volume!r}."
            )

    def get_embedder(self, key: str, **kwargs: Any) -> TextEmbedder:
        model_name = kwargs.get("model_name", _key_to_model_name(key))
        output_dim = _resolve_output_dim(model_name)
        return ModalEmbedder(
            model_name=model_name,
            output_dim=output_dim,
            gpu_cls=ModalEmbeddingGPU,
        )

    def run_preprocess(self, job: PreprocessJob) -> DataSource:
        logger.info(f"Dispatching preprocess job to Modal project {self._project!r}")
        return run_preprocess_remote.remote(job)


def _key_to_model_name(key: str) -> str:
    """Map a local embedder registry key to its HuggingFace model name."""
    mapping = {
        "modernbert-base": "answerdotai/ModernBERT-base",
    }
    if key not in mapping:
        raise ValueError(
            f"Modal runtime does not support embedder {key!r}. "
            f"Supported keys: {list(mapping)}"
        )
    return mapping[key]


def _resolve_output_dim(model_name: str) -> int:
    """Return the embedding dimension for known models without loading them."""
    if model_name in {"answerdotai/ModernBERT-base", "modernbert-base"}:
        return 768
    raise ValueError(
        f"Unknown output dimension for {model_name!r}. "
        "Add it to _resolve_output_dim in src/runtime/modal_runtime.py."
    )


Runtime.register(ModalRuntime)
