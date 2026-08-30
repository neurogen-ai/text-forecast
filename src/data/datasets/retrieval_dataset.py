"""Datasets for embedding-retrieval forecasting.

- :class:`VectorStoreDataset` loads a precomputed embedding column from a
  parquet dataset into an in-memory vector database, ready to be queried by
  the model's query embedder (FAISS when available, torch matmul otherwise).
- :class:`RetrievalDataset` serves the target papers themselves (token ids +
  binary threshold target) exactly like :class:`TextTokenDataset`; it exists
  as a distinct class so retrieval runs are identifiable in configs and logs.
"""

from __future__ import annotations

from logging import getLogger
from pathlib import Path

import polars as pl
import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.utils.data import Dataset  # type: ignore[reportUnknownVariableType]

from data.datasets.text_token_dataset import (
    TextTokenDataset,
    TextTokenDatasetConfig,
)
from data.sources import DataSource
from utils import component
from utils.logging import setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore[import-untyped]

    _HAS_FAISS = True
except ImportError:
    faiss = None
    _HAS_FAISS = False


class VectorStoreDatasetConfig(BaseModel):
    loc: str
    embedding_col: str
    filter: pl.Expr | None = None
    max_rows: int | None = None
    normalize: bool = True
    name: str
    model_config = ConfigDict(arbitrary_types_allowed=True)


@component
class VectorStoreDataset(Dataset):
    """In-memory vector database built from a parquet embedding column.

    Loads ``config.embedding_col`` (a list-of-floats column produced by the
    preprocess embed step) from every parquet file under the source, stacks
    it into an ``(M, D)`` tensor, and optionally L2-normalises it for inner
    product search. Exposes ``search`` for no-grad top-k retrieval plus the
    raw ``db`` tensor for differentiable re-scoring in the model.
    """

    config = VectorStoreDatasetConfig

    def __init__(
        self,
        *,
        config: VectorStoreDatasetConfig,
        source: DataSource,
    ) -> None:
        super().__init__()
        self.config = config
        data_path = Path(source.resolve())
        files = list(data_path.glob("*.par*"))
        if not files:
            raise FileNotFoundError(f"No parquet files under {data_path}")

        lf = pl.scan_parquet(files)
        needed = [config.embedding_col]
        if config.filter is not None:
            # The predicate may reference columns beyond the embedding column.
            needed += [c for c in config.filter.meta.root_names() if c not in needed]
        lf = lf.select(needed)
        lf = lf.drop_nulls(config.embedding_col)
        if config.filter is not None:
            lf = lf.filter(config.filter)
        df = lf.collect(engine="streaming").select(config.embedding_col)
        if config.max_rows is not None and len(df) > config.max_rows:
            df = df.sample(n=config.max_rows, shuffle=True, seed=0)

        db = torch.tensor(
            df[config.embedding_col].to_list(), dtype=torch.float32
        )
        if config.normalize:
            db = torch.nn.functional.normalize(db, dim=-1)
        self.db: Tensor = db
        logger.info(f"{config.name}: vector db {tuple(db.shape)}")

        self.faiss_index = None
        if _HAS_FAISS:
            index = faiss.IndexFlatIP(db.size(-1))
            index.add(db.numpy())
            self.faiss_index = index
            logger.info(f"{config.name}: FAISS index built")

    def __len__(self) -> int:
        return self.db.size(0)

    @property
    def dim(self) -> int:
        return self.db.size(-1)

    @torch.no_grad()
    def search(self, query: Tensor, top_k: int) -> Tensor:
        """No-grad top-k retrieval. (N, D) or (..., D) -> (..., top_k) ids."""
        if self.faiss_index is not None:
            flat = query.detach().reshape(-1, self.dim).cpu().numpy()
            _, idx = self.faiss_index.search(flat, top_k)
            return torch.from_numpy(idx).reshape(*query.shape[:-1], top_k)
        scores = query.detach().float() @ self.db.T
        k = min(top_k, scores.size(-1))
        _, idx = torch.topk(scores, k=k, dim=-1)
        return idx


@component
class RetrievalDataset(TextTokenDataset):
    """Token-id dataset whose examples feed the retrieval-forecast model.

    Behaviour is identical to ``TextTokenDataset`` (concatenated token
    columns, binary threshold target at ``config.theta``, ``TokenBatch``
    output).
    """

    config = TextTokenDatasetConfig
