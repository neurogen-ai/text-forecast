"""Modal runtime: volumes, image, GPU embedder, and remote preprocess dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

from config.env import Env
from data.preprocess.pipeline import PreprocessJob, run_preprocess_pipeline
from data.sources.base import DataSource
from data.sources.modal import ModalVolumeSource
from runtime.base import Runtime, TextEmbedder

preprocess_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("src", remote_path="/root/src")
    .env({"PYTHONPATH": "/root/src"})
)

app = modal.App("citef-preprocess")


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
