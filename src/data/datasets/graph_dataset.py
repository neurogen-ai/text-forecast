import os
import shutil
from collections.abc import Mapping
from datetime import date
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, Self, override

import polars as pl
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.utils.data import Dataset

from data.formaters import Formater, GraphFormater
from data.sources import DataSource
from utils import component
from utils.logging import setup_logger

from .polars_dataset import plot_target_distribution_polars
from .types import CitationGraphDatasetOutput

logger = getLogger(__name__)
_ = setup_logger(logger)


class CitationGraphDatasetConfig(BaseModel):
    loc: str
    x: list[str]
    y: list[str]
    meta_cols: list[str]
    filter: pl.Expr | None
    weights: Tensor | None
    max_len: int
    graph_max_len: int
    shuffle: bool = True
    sample: int | None
    return_mask: bool
    pad: bool
    pad_value: Any
    truncate: bool
    truncate_method: Literal["truncate", "drop"]
    name: str
    auto_remove: bool
    time_col: str
    t_start: date | None
    t_end: date | None
    return_id: bool
    id_col: str = "id"
    subsample: int | None
    category_cols: list[str]
    sort_cols: list[str]
    top_k: int
    add_x: list[str]
    max_mem_rows: int
    model_config = ConfigDict(arbitrary_types_allowed=True)


@component
class GraphDataset[T_Config](Dataset[CitationGraphDatasetOutput]):
    config = CitationGraphDatasetConfig

    def __init__(
        self,
        *,
        config: CitationGraphDatasetConfig,
        source: DataSource,
        formater: Formater[Mapping[str, Any], CitationGraphDatasetOutput] | None = None,
    ):
        super().__init__()

        assert not (config.weights is None and not config.y), (
            "Weights cannot be given when no y is given"
        )
        assert not (config.shuffle and config.sample is not None), (
            "Shuffle and Sample cannot both be used"
        )
        self.config = config
        self.source = source
        self.formater = formater or GraphFormater(
            max_len=config.max_len,
            graph_max_len=config.graph_max_len,
            top_k=config.top_k,
            pad=config.pad,
            truncate=config.truncate,
            truncate_method=config.truncate_method,
            return_mask=config.return_mask,
            return_id=config.return_id,
            weights=config.weights,
        )
        self.data_path = source.resolve()
        self.meta_cols = config.meta_cols
        self.filter = config.filter
        self.t_start = config.t_start
        self.t_end = config.t_end
        self.weights = config.weights
        self.time_col = config.time_col
        self.return_id = config.return_id
        self.id_col = config.id_col
        self.max_len = config.max_len
        self.graph_max_len = config.graph_max_len
        self.pad = config.pad
        self.pad_value = config.pad_value
        self.return_mask = config.return_mask
        self.truncate = config.truncate
        self.truncate_method = config.truncate_method
        self.name = config.name
        self.hot_path: Path = Path("./.temp") / "hot" / self.name
        self.subsample = config.subsample
        self.x = config.x
        self.y = config.y
        self.add_x = config.add_x

        self.category_cols = config.category_cols
        self.sort_cols = config.sort_cols
        self.top_k = config.top_k

        if self.subsample is not None:
            self.hot_path = Path("./.temp") / "hot" / f"{self.name}-DRY"

        x_columns = list(
            set(
                self.x
                + self.meta_cols
                + self.category_cols
                + self.sort_cols
                + self.add_x
                + [self.time_col]
            )
        )
        columns = list(set(x_columns + self.y + [config.id_col, "referenced_works"]))
        self.x_hot_path: Path = self.hot_path / "x.ipc"
        self.add_x_hot_path: Path = self.hot_path / "add_x.ipc"
        self.y_hot_path: Path = self.hot_path / "y.ipc"
        self.id_hot_path: Path = self.hot_path / "id.ipc"

        if self.hot_path.exists() and config.auto_remove:
            logger.info(f"Found data at {self.hot_path}, deleting")
            shutil.rmtree(self.hot_path)

        files = list(Path(self.data_path).glob("*.par*"))

        lf: pl.LazyFrame = pl.scan_parquet(files).select(columns)

        lf = lf.drop_nulls(columns)

        if self.filter is not None:
            lf = lf.filter(self.filter)

        if self.t_end is not None and config.time_col:
            lf = lf.filter(pl.col(self.time_col) < self.t_end)


                # Construct the conditional expression by iterating in reverse
        final_sort_expr = pl.lit(None)
        for idx, col in reversed(list(enumerate(config.category_cols))):
            final_sort_expr = pl.when(
                pl.col(col) == pl.col(f"{col}_referenced")
            ).then(idx).otherwise(final_sort_expr)

        lf = lf.with_columns(
                    graph_x_single=pl.concat_list(self.add_x),
                )

        graph_x_df = (
            lf.explode("referenced_works")
            .join(
                lf.select(
                    [self.id_col, "graph_x_single"]
                    + self.category_cols
                    + self.sort_cols
                ),
                left_on="referenced_works",
                right_on=self.id_col,
                suffix="_referenced",
            )
            .filter(
                pl.any_horizontal(
                    [
                        pl.col(cat_col) == pl.col(f"{cat_col}_referenced")
                        for cat_col in self.category_cols
                    ]
                )
            )
            .with_columns(
                final_sort_col=final_sort_expr
            )
            .group_by(self.id_col)
            .agg(
                [
                    pl.col("graph_x_single_referenced")
                    .sort_by(['final_sort_col'] + self.sort_cols)
                    .head(self.top_k)
                    .list.explode()
                    .alias("graph_x")
                ]
            )
        )

        graph_x_df = graph_x_df.filter(pl.col("graph_x").list.len() >= 1)
        lf = (
            lf.join(graph_x_df.lazy(), on=self.id_col, how="inner")
            .with_columns(
                x=pl.concat_list(self.x),
                y=pl.concat_list(self.y),
            )
            .drop_nulls(["graph_x", "x", "y"])
        )

        if self.t_start is not None and config.time_col:
            lf = lf.filter((pl.col(self.time_col) >= self.t_start))

        if self.truncate:
            over_x = pl.col("x").list.len() > self.max_len
            over_g = pl.col("graph_x").list.len() > self.graph_max_len
            over = over_x | over_g
            n_over: int | None = lf.select(over.sum()).collect(engine="streaming").item()
            if self.truncate_method == "drop":
                logger.info(
                    f"{self.name}: dropping {n_over} rows where x > {self.max_len} or graph_x > {self.graph_max_len}"
                )
                lf = lf.filter(~over)
            else:
                logger.info(
                    f"{self.name}: {n_over} rows exceed max_len/graph_max_len and will be truncated per item"
                )

        if self.subsample is not None:
            logger.info(f"Taking {self.subsample} subsamples")
            lf = lf.slice(0, self.subsample)

        rows = lf.select(pl.len()).collect(engine="streaming").item()
        logger.info(
            f"{rows:,} rows where x len <= {self.max_len} & total graph length <= {self.graph_max_len} to hotpath: {self.name}"
        )
        if rows > config.max_mem_rows:
            os.makedirs(self.hot_path, exist_ok=True)
            lf.select(["x", "graph_x", "y", "id"]).sink_ipc(self.x_hot_path)

        if rows is None or rows > config.max_mem_rows:
            self.df: pl.DataFrame = pl.read_ipc(self.x_hot_path, memory_map=True)
            logger.info(f"Hot path {self.hot_path} loaded")
        else:
            self.df = lf.collect(engine="streaming")
            logger.info(f"Hot path {self.hot_path} loaded into mem")

    def with_window(
        self,
        *,
        t_start: date | None = None,
        t_end: date | None = None,
    ) -> Self:
        """Return a new dataset over the same source with updated time bounds.

        This re-initialises the dataset from its stored config so that the
        in-memory Polars frame reflects the new window (D1).
        """
        new_config = self.config.model_copy(
            update={"t_start": t_start, "t_end": t_end}
        )
        return type(self)(
            config=new_config,
            source=self.source,
            formater=self.formater,
        )

    def __len__(self) -> int:
        return len(self.df)

    @override
    def __getitem__(self, idx: int) -> CitationGraphDatasetOutput:
        row = self.df.row(idx, named=True)
        return self.formater(row)
