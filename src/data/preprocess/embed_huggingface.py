"""Local HuggingFace embedding implementation with lazy model loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, NamedTuple

import torch
from transformers import AutoModel, AutoTokenizer

from utils import component

from .embed import TextEmbedder, register_embedder

logger = getLogger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

class EmbedderConfig(NamedTuple):
    model_name: str
    output_dim: int
    bos_token: str
    eos_token: str

EMBEDDERS: dict[str, EmbedderConfig] = {
    "modernbert-base": EmbedderConfig(
        model_name="answerdotai/ModernBERT-base",
        output_dim=768,
        bos_token='[CLS]',
        eos_token='[SEP]',
    ),
    "modernbert-embed-base": EmbedderConfig(
        model_name="nomic-ai/modernbert-embed-base",
        output_dim=768,
        bos_token='[CLS]',
        eos_token='[SEP]',
    ),
}

@register_embedder("modernbert-base")
@component
@dataclass(kw_only=True)
class HuggingFaceEmbedder:
    """Lazy-loading HuggingFace sentence encoder with mean pooling."""

    model_name: str = "answerdotai/ModernBERT-base"
    device: str | None = None
    batch_size: int = 32
    dtype: str = "bfloat16"
    pooling: str = "mean"
    compile: bool = False
    cache_dir: str | None = None

    _tokenizer: Any = field(init=False, repr=False, default=None)
    _model: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.pooling != "mean":
            raise ValueError(f"Unsupported pooling {self.pooling!r}")
        if self.dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype {self.dtype!r}; choose from {list(_DTYPE_MAP)}"
            )

    def load(self) -> None:
        """Eagerly load the tokenizer and model into memory."""
        logger.info("HuggingFaceEmbedder.load: loading model %r", self.model_name)
        self._ensure_loaded()
        logger.info("HuggingFaceEmbedder.load: model %r loaded", self.model_name)

    def _resolve_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None:
            logger.debug(
                "HuggingFaceEmbedder._ensure_loaded: model %r already loaded",
                self.model_name,
            )
            return

        logger.info(
            "HuggingFaceEmbedder._ensure_loaded: loading tokenizer for %r",
            self.model_name,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        logger.info(
            "HuggingFaceEmbedder._ensure_loaded: loading model %r (dtype=%s)",
            self.model_name,
            self.dtype,
        )
        torch_dtype = _DTYPE_MAP[self.dtype]
        self._model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            reference_compile=False,
            cache_dir=self.cache_dir,
        )
        logger.debug("HuggingFaceEmbedder._ensure_loaded: moving model to device")
        self._model.eval()
        target_device = self._resolve_device()
        self._model = self._model.to(target_device)
        if self.compile:
            logger.debug("HuggingFaceEmbedder._ensure_loaded: compiling model")
            self._model.compile(mode="default")
        logger.info(
            "HuggingFaceEmbedder._ensure_loaded: model %r ready on %s",
            self.model_name,
            target_device,
        )

    @property
    def output_dim(self) -> int:
        self._ensure_loaded()
        return int(self._model.config.hidden_size)

    def encode(self, texts: list[str]) -> list[list[float]]:
        logger.debug(
            "HuggingFaceEmbedder.encode: starting batch of %d texts",
            len(texts),
        )
        self._ensure_loaded()
        device = self._resolve_device()
        torch_dtype = _DTYPE_MAP[self.dtype]

        unk_token = self._tokenizer.unk_token or " "
        safe_texts = [t if t.strip() else unk_token for t in texts]

        inputs = self._tokenizer(
            safe_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        token_embeddings = outputs.last_hidden_state.to(torch_dtype)
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .to(torch_dtype)
        )

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)

        # Guard padding-only / empty samples by zeroing their contribution.
        zero_mask = (sum_mask == 0).expand_as(sum_embeddings)
        sum_embeddings = torch.where(
            zero_mask, torch.zeros_like(sum_embeddings), sum_embeddings
        )
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        embeddings = sum_embeddings / sum_mask
        result = embeddings.float().cpu().tolist()
        logger.debug(
            "HuggingFaceEmbedder.encode: finished batch of %d texts -> %d embeddings",
            len(texts),
            len(result),
        )
        return result
