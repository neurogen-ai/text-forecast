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
    time_col: str | None = "publication_date"
    delta_years: float = 0.0
    name: str
    model_config = ConfigDict(arbitrary_types_allowed=True)


@component
class VectorStoreDataset(Dataset):
    """In-memory vector database built from a parquet embedding column.

    Loads ``config.embedding_col`` (a list-of-floats column produced by the
    preprocess embed step) from every parquet file under the source, stacks
    it into an ``(M, D)`` tensor, and optionally L2-normalises it for inner
    product search. Exposes ``search`` for no-grad top-k retrieval plus the
    raw ``db`` tensor for differentiable re-scoring in the model. When
    ``config.time_col`` is set, publication dates are also loaded and exposed
    as ``dates`` (float32 days-since-epoch ordinals) so the model can apply
    the ``delta_years`` recency filter during vector search: a database row
    is only retrievable for an example whose publication date is at least
    ``delta_years`` years later (rows with date greater than
    ``example_date - delta_years * 365.25`` days are excluded; the boundary
    itself is inclusive).
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
        if config.time_col is not None and config.time_col not in needed:
            needed.append(config.time_col)
        if config.filter is not None:
            lf = lf.filter(config.filter)

        lf = lf.select(needed)
        lf = lf.drop_nulls(needed)

        df = lf.collect(engine="streaming").select(needed)
        if config.max_rows is not None and len(df) > config.max_rows:
            df = df.sample(n=config.max_rows, shuffle=True, seed=0)

        db = torch.tensor(
            df[config.embedding_col].to_list(), dtype=torch.float32
        )
        if config.normalize:
            db = torch.nn.functional.normalize(db, dim=-1)
        self.db: Tensor = db
        logger.info(f"{config.name}: vector db {tuple(db.shape)}")

        self.dates: Tensor | None = None
        if config.time_col is not None:
            date_col = df[config.time_col]
            if not date_col.dtype.is_temporal():
                raise ValueError(
                    f"{config.name}: time column {config.time_col!r} has "
                    f"non-temporal dtype {date_col.dtype}; expected a Date or "
                    "Datetime column"
                )
            # Polars Date's physical repr is days since the Unix epoch
            # (1970-01-01 -> 0). This is the same unit the token datasets
            # emit for ``TokenBatch.date`` (also epoch days), so store and
            # batch dates compare directly inside the retriever.
            physical = date_col.cast(pl.Date, strict=False).to_physical()
            self.dates = torch.tensor(physical.to_list(), dtype=torch.float32)
        logger.info(f"{config.name}: delta_years={config.delta_years}")

        self.faiss_index = None
        if faiss is not None and _HAS_FAISS:
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
        """Unfiltered no-grad top-k retrieval. (N, D) or (..., D) -> (..., top_k).

        Recency filtering (``delta_years``) does not happen here: with dates,
        the model's :class:`~models.retrieval_forecast.VectorRetriever` runs
        a masked top-k against its own device-local buffer, which keeps one
        exact code path and avoids faiss-version-dependent search params.
        """
        if self.faiss_index is not None:
            flat = query.detach().reshape(-1, self.dim).cpu().numpy()
            _, idx = self.faiss_index.search(flat, top_k)
            return torch.from_numpy(idx).reshape(*query.shape[:-1], top_k)
        query = query.detach().to(self.db.device)
        scores = query.float() @ self.db.T
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
