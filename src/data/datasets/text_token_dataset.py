"""Simple token-id text dataset for plain sequence classification.

Unlike ``GraphDataset`` this dataset does **not** build a citation-graph
neighbourhood; it only concatenates the requested token columns (e.g.
``title abstract_tokens``) and returns a ``TokenBatch`` suitable for
``TransformerClass`` and the binary classification strategy.
"""

from __future__ import annotations

from datetime import date, datetime
from logging import getLogger
from pathlib import Path
from typing import Self, override

import polars as pl
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.utils.data import Dataset, default_collate  # type: ignore[reportUnknownVariableType]

from data.datasets.presence import assert_dataset_present
from data.datasets.truncation import TruncateMethod, apply_truncation_policy
from data.datasets.types import TokenBatch
from data.sources import DataSource
from utils import component
from utils.logging import setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)


class TextTokenDatasetConfig(BaseModel):
    loc: str
    x: list[str]
    y: list[str]
    id_col: str = "id"
    meta_cols: list[str] = []
    filter: pl.Expr | None = None
    weights: Tensor | None = None
    max_len: int
    pad_token_id: int
    pad: bool = True
    truncate: bool = True
    truncate_method: TruncateMethod = "truncate"
    name: str
    auto_remove: bool = True
    time_col: str | None = "publication_date"
    t_start: date | None = None
    t_end: date | None = None
    return_id: bool = True
    subsample: int | None = None
    theta: float = 0.0
    return_date: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)


@component
class TextTokenDataset(Dataset[TokenBatch]):
    """Token-id dataset for text classification.

    Assumes the parquet source already contains tokenised list columns
    (e.g. ``title abstract_tokens`` produced by the preprocess tokenise step).
    The target column is treated as a scalar count and thresholded at
    ``config.theta`` for binary classification.
    """

    config = TextTokenDatasetConfig

    def __init__(
        self,
        *,
        config: TextTokenDatasetConfig,
        source: DataSource,
    ) -> None:
        super().__init__()
        self.config = config
        self.source = source
        self.data_path = assert_dataset_present(source, config.name)
        self.max_len = config.max_len
        self.pad_value = config.pad_token_id
        self.pad = config.pad
        self.truncate = config.truncate
        self.name = config.name
        self.return_id = config.return_id
        self.id_col = config.id_col
        self.return_date = config.return_date
        self.time_col = config.time_col
        if self.return_date and config.time_col is None:
            raise ValueError(
                f"{config.name}: return_date=True requires time_col to be set"
            )
        self.subsample = config.subsample
        self.weights = config.weights
        self.theta = config.theta

        columns = list(
            set(
                config.x
                + config.y
                + config.meta_cols
                + ([config.id_col] if config.return_id else [])
            )
        )
        if config.time_col:
            columns.append(config.time_col)

        files = list(Path(self.data_path).glob("*.par*"))
        lf: pl.LazyFrame = pl.scan_parquet(files).select(columns)

        lf = lf.drop_nulls(columns)

        if config.filter is not None:
            lf = lf.filter(config.filter)

        if config.t_start is not None and config.time_col:
            lf = lf.filter(pl.col(config.time_col) >= config.t_start)
        if config.t_end is not None and config.time_col:
            lf = lf.filter(pl.col(config.time_col) < config.t_end)

        # The time column stays in the frame when return_date is set, so
        # __getitem__ can emit per-example publication dates for retrieval
        # models that filter their vector search by date (delta filter).
        if config.time_col and not self.return_date:
            lf = lf.drop(config.time_col)
        if config.meta_cols:
            lf = lf.drop(config.meta_cols)

        lf = lf.with_columns(
            x=pl.concat_list(config.x),
            y=pl.concat_list(config.y),
        )

        lf = apply_truncation_policy(
            lf,
            name=self.name,
            limits={"x": self.max_len},
            truncate=self.truncate,
            method=config.truncate_method,
        )

        if self.subsample is not None:
            logger.info(f"{self.name}: taking {self.subsample} subsamples")
            lf = lf.collect(engine="streaming").sample(n=self.subsample).lazy()

        self.df = lf.collect(engine="streaming")
        logger.info(f"{self.name}: loaded {len(self.df)} rows")

    def with_window(
        self,
        *,
        t_start: date | None = None,
        t_end: date | None = None,
    ) -> Self:
        """Return a new dataset over the same source with updated time bounds."""
        new_config = self.config.model_copy(
            update={"t_start": t_start, "t_end": t_end}
        )
        return type(self)(config=new_config, source=self.source)

    def __len__(self) -> int:
        return len(self.df)

    @override
    def __getitem__(self, idx: int) -> TokenBatch:
        row = self.df.row(idx, named=True)
        x: Tensor = torch.tensor(row["x"], dtype=torch.long)
        y_raw: Tensor = torch.tensor(row["y"], dtype=torch.float32).flatten()
        y = (y_raw > self.theta).float()

        if self.pad and x.size(0) < self.max_len:
            x = nn.functional.pad(
                x, (0, self.max_len - x.size(0)), value=self.pad_value
            )

        if self.truncate and x.size(0) > self.max_len:
            x = x[: self.max_len]

        # Plain (T,) key-padding mask; collated to (B, T). Attention layers
        # broadcast to head shape themselves (plan 1.4.1).
        mask = (x != self.pad_value).bool()

        id_out = torch.tensor(float("nan"))
        if self.return_id:
            id_out = torch.tensor(row[self.id_col])

        date_out: Tensor | None = None
        if self.return_date:
            assert self.time_col is not None  # validated in __init__
            raw_date = row[self.time_col]
            if raw_date is None:
                raise ValueError(
                    f"{self.name}: null {self.time_col} at row {idx}; "
                    "return_date=True requires a publication date"
                )
            # Polars Date rows arrive as python ``date`` (Datetime as
            # ``datetime``). Emit days since the Unix epoch so the unit
            # matches the vector store's date encoding exactly.
            if isinstance(raw_date, datetime):
                raw_date = raw_date.date()
            date_out = torch.tensor(
                float((raw_date - date(1970, 1, 1)).days), dtype=torch.float32
            )

        # Optional weights (sub-plan 1.4.6): None when the dataset has no
        # weights configured, a scalar ``(B,)``-collated tensor otherwise.
        weight: Tensor | None = None
        if self.weights is not None:
            weight = torch.tensor(
                self.weights[y.long()].item(), dtype=torch.float32
            )

        return TokenBatch(
            id=id_out,
            x=x,
            y=y,
            mask=mask,
            weight=weight,
            date=date_out,
        )


def token_batch_collate(batches: list[TokenBatch]) -> TokenBatch:
    """Collate a list of per-item ``TokenBatch`` into a single batch.

    Needed because ``default_collate`` cannot stack a ``None`` weight field.
    Tensor fields use the default collator; ``weight`` stays ``None`` when the
    dataset emits none, else stacks into ``(B,)``.
    """
    weight: Tensor | None = batches[0].weight
    if weight is not None:
        weight = default_collate([b.weight for b in batches])  # type: ignore[arg-type]

    date_field: Tensor | None = batches[0].date
    if date_field is not None:
        date_field = default_collate([b.date for b in batches])  # type: ignore[arg-type]

    return TokenBatch(
        id=default_collate([b.id for b in batches]),  # type: ignore[arg-type]
        x=default_collate([b.x for b in batches]),  # type: ignore[arg-type]
        y=default_collate([b.y for b in batches]),  # type: ignore[arg-type]
        mask=default_collate([b.mask for b in batches]),  # type: ignore[arg-type]
        weight=weight,
        date=date_field,
    )
