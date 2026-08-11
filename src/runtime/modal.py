"""Modal runtime: volumes, image, GPU embedder, and remote preprocess dispatch."""

from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any

import modal

from config.env import Env
from data.preprocess.embed_modal import ModalEmbedder
from data.preprocess.pipeline import PreprocessJob, run_preprocess_pipeline
from data.sources.base import DataSource
from data.sources.modal import ModalVolumeSource
from runtime.base import Runtime, TextEmbedder

logger = getLogger(__name__)

preprocess_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("src", remote_path="/root/src")
    .env({"PYTHONPATH": "/root/src"})
)

app = modal.App("citef-preprocess")

# Volume labels mounted by the remote preprocess function.  These are module
# constants because Modal function decorators are evaluated at import time; in
# v1.2 the config values under [runtime.modal] must match them.
RAW_VOLUME_LABEL = "openalex-raw"
STAGED_VOLUME_LABEL = "openalex-staged"

raw_volume = modal.Volume.from_name(RAW_VOLUME_LABEL, create_if_missing=True)
staged_volume = modal.Volume.from_name(STAGED_VOLUME_LABEL, create_if_missing=True)


@app.cls(
    gpu="T4",
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
    volumes={
        f"/modal/{RAW_VOLUME_LABEL}": raw_volume,
        f"/modal/{STAGED_VOLUME_LABEL}": staged_volume,
    },
    image=preprocess_image,
)
def run_preprocess_remote(job: PreprocessJob) -> DataSource:
    """Container entry point for the full preprocessing pipeline.

    Volumes are mounted at the same paths ``ModalVolumeSource.resolve()``
    returns, so a lightweight ``LocalRuntime`` can run the pipeline unchanged.
    """
    from runtime.local import LocalRuntime

    local_runtime = LocalRuntime()
    return run_preprocess_pipeline(job, local_runtime)


class ModalRuntime:
    """Modal backend: volume sources, GPU embedder, and remote job dispatch."""

    def __init__(
        self,
        env: Env,
        project: str,
        raw_volume: str,
        staged_volume: str,
        gpu: str,
        embedder_batch_size: int,
    ) -> None:
        self._env = env
        self._project = project
        self._gpu_type = gpu
        self._embedder_batch_size = embedder_batch_size

        if raw_volume != RAW_VOLUME_LABEL or staged_volume != STAGED_VOLUME_LABEL:
            raise ValueError(
                "Modal volume labels in config/config.toml must match the "
                f"module defaults in v1.2: raw_volume={RAW_VOLUME_LABEL!r}, "
                f"staged_volume={STAGED_VOLUME_LABEL!r}."
            )

    def _is_raw_root(self, root: str | Path) -> bool:
        return Path(root) == Path(self._env.raw_loc)

    def get_source(self, root: str | Path, name: str) -> DataSource:
        volume_label = (
            RAW_VOLUME_LABEL if self._is_raw_root(root) else STAGED_VOLUME_LABEL
        )
        return ModalVolumeSource(volume_label=volume_label, path="", name=name)

    def get_embedder(self, key: str, **kwargs: Any) -> TextEmbedder:
        model_name = kwargs.get("model_name", _key_to_model_name(key))
        output_dim = _resolve_output_dim(model_name)
        return ModalEmbedder(
            model_name=model_name,
            output_dim=output_dim,
            gpu_cls=ModalEmbeddingGPU,
        )

    def maybe_download(self, source: DataSource) -> Path:
        return source.resolve()

    def maybe_upload(self, source: DataSource, local_path: Path) -> None:
        return None

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
        "Add it to _resolve_output_dim in src/runtime/modal.py."
    )


Runtime.register(ModalRuntime)
