import os
import shutil
from datetime import date
from logging import getLogger
from pathlib import Path
from typing import Literal, NamedTuple, Protocol, override

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.utils.data import Dataset

from utils import component
from utils.logging import setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)


class PolarsDatasetConfig(BaseModel):
    loc: str
    x: list[str]
    y: list[str]
    meta_cols: list[str]
    filter: pl.Expr | None
    weights: Tensor | None
    max_len: int
    shuffle: bool = True
    sample: int | None
    return_mask: bool
    pad: bool
    pad_token_id: int
    truncate: bool
    truncate_method: Literal["truncate", "drop"]
    name: str
    auto_remove: bool
    time_col: str
    t_start: date | None
    t_end: date | None
    return_id: bool
    id_col: str
    subsample: int | None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Env(Protocol):
    STAGED_LOC: Path


class PolarsDatasetOutput(NamedTuple):
    id: Tensor
    x: Tensor
    y: Tensor
    weight: Tensor
    mask: Tensor


def plot_target_distribution_polars(lf: pl.LazyFrame, y_true_col: str, log_scale=True):
    """Plots a histogram of the true target values from a Polars LazyFrame."""
    # Only evaluate and collect the single target column to save memory
    y_true = lf.select(y_true_col).collect(engine="streaming").to_numpy()

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(9, 5))

    sns.histplot(
        y_true + 1,
        kde=True,
        color="#1f77b4",
        edgecolor="white",
        linewidth=1.2,
        alpha=0.6,
        bins="auto",
        log_scale=log_scale,
    )

    plt.title(
        f"Distribution of True Target Values ({y_true_col})",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("True Target Value", fontsize=12, labelpad=10)
    plt.ylabel("Count / Density", fontsize=12, labelpad=10)
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    return fig


@component
class PolarsDataset[T_Config](Dataset[PolarsDatasetOutput]):
    config = PolarsDatasetConfig

    def __init__(
        self,
        config: PolarsDatasetConfig,
        env: Env,
    ):
        super().__init__()

        assert not (config.weights is None and not config.y), (
            "Weights cannot be given when no y is given"
        )
        assert not (config.shuffle and config.sample is not None), (
            "Shuffle and Sample cannot both be used"
        )
        self.data_path = env.STAGED_LOC / config.loc
        self.meta_cols = config.meta_cols
        self.filter = config.filter
        self.t_start = config.t_start
        self.t_end = config.t_end
        self.weights = config.weights
        self.time_col = config.time_col
        self.return_id = config.return_id
        self.id_col = config.id_col
        self.max_len = config.max_len
        self.pad = config.pad
        self.pad_value = config.pad_token_id
        self.return_mask = config.return_mask
        self.truncate = config.truncate
        self.truncate_method = config.truncate_method
        self.name = config.name
        self.hot_path: Path = Path("./.temp") / "hot" / self.name
        self.subsample = config.subsample
        if self.subsample is not None:
            self.hot_path = Path("./.temp") / "hot" / f"{self.name}-DRY"

        if isinstance(config.x, str):
            self.x: list[str] = [config.x]
        else:
            self.x = config.x
        if isinstance(config.y, str):
            self.y: list[str] = [config.y]
        else:
            self.y = config.y
        columns = list(set(self.x + self.y + self.meta_cols))

        if config.time_col:
            columns += [config.time_col]
        if config.return_id:
            columns += [config.id_col]

        self.x_hot_path: Path = self.hot_path / "x.ipc"
        self.y_hot_path: Path = self.hot_path / "y.ipc"
        self.id_hot_path: Path = self.hot_path / "id.ipc"

        if self.hot_path.exists() and config.auto_remove:
            logger.info(f"Found data at {self.hot_path}, deleting")
            shutil.rmtree(self.hot_path)
        if not self.hot_path.exists():
            os.makedirs(self.hot_path, exist_ok=True)
            files = list(Path(self.data_path).glob("*.par*"))

            lf: pl.LazyFrame = pl.scan_parquet(files).select(columns)

            if self.t_start is not None and config.time_col:
                lf = lf.filter((pl.col(self.time_col) >= self.t_start))
            if self.t_end is not None and config.time_col:
                lf = lf.filter(pl.col(self.time_col) < self.t_end)

            lf = lf.drop_nulls(columns)

            if self.truncate:
                lf = lf.with_columns(total_len=pl.lit(0, dtype=pl.Int32))

                for col in self.x:
                    lf = lf.with_columns(
                        total_len=pl.col("total_len") + pl.col(col).list.len()
                    )
                over = pl.col("total_len") > self.max_len
                n_over: int | None = (
                    lf.select(over.sum()).collect(engine="streaming").item()
                )
                if self.truncate_method == "drop":
                    logger.info(
                        f"{self.name}: dropping {n_over} rows with combined x length > {self.max_len}"
                    )
                    lf = lf.filter(~over)
                else:
                    logger.info(
                        f"{self.name}: {n_over} rows exceed max_len={self.max_len} and will be truncated per item"
                    )
                lf = lf.drop("total_len")

            if self.filter is not None:
                lf = lf.filter(self.filter)

            if self.time_col:
                lf = lf.drop([self.time_col])
            if self.meta_cols:
                lf = lf.drop(self.meta_cols)

            if self.subsample is not None:
                logger.info(f"Taking {self.subsample} random subsamples")
                lf = lf.collect(engine="streaming").sample(n=self.subsample).lazy()


            rows = lf.select(pl.len()).collect(engine="streaming").item()
            logger.info(
                f"Saving {rows:,} rows where summed length of {self.x} <= {self.max_len} to hotpath: {self.name}"
            )

            lf.select(self.x).sink_ipc(self.x_hot_path)
            lf.select(self.y).sink_ipc(self.y_hot_path)
            if self.return_id:
                lf.select([self.id_col]).sink_ipc(self.id_hot_path)

        self.df_x: pl.DataFrame = pl.read_ipc(self.x_hot_path, memory_map=True)
        self.df_y: pl.DataFrame = pl.read_ipc(self.y_hot_path, memory_map=True)
        if self.return_id:
            self.df_id: pl.DataFrame = pl.read_ipc(self.id_hot_path, memory_map=True)
        logger.info(f"Hot path {self.hot_path} loaded")

    def __len__(self) -> int:
        return len(self.df_x)

    def _format_x(self, x: Tensor) -> Tensor:
        return x.long()

    def _format_y(self, y: Tensor) -> Tensor:
        return y

    @override
    def __getitem__(self, idx: int) -> PolarsDatasetOutput:
        id = torch.tensor(float("nan"))
        x = torch.tensor(float("nan"))
        y = torch.tensor(float("nan"))
        mask = torch.tensor(float("nan"))
        weight = torch.tensor(float("nan"))
        # X (input)
        x_row: tuple[list[int], ...] = self.df_x.row(idx)
        x: Tensor = torch.cat(
            [torch.tensor(token_list) for token_list in x_row]
        ).flatten()

        x = self._format_x(x)
        if self.pad and x.size(0) < self.max_len:
            x = nn.functional.pad(
                x, (0, self.max_len - x.size(0)), value=self.pad_value
            )
        if self.truncate == True & x.size(0) > self.max_len:
            x = x[: self.max_len]

        # y (target)
        if self.y:
            y_row: tuple[list[int], ...] = self.df_y.row(idx)
            y: Tensor = torch.stack(
                [torch.tensor(target, dtype=torch.float32) for target in y_row]
            ).flatten()
            y = self._format_y(y)

        # id (tracking)
        if self.return_id:
            id = torch.tensor(self.df_id.row(idx))

        # weights (per target class)
        if self.weights is not None:
            weight: Tensor = torch.tensor(
                [
                    torch.tensor(self.weights[target.long()], dtype=torch.float32)
                    for target in torch.atleast_1d(y)
                ]
            ).flatten()

        if self.return_mask:
            mask = (x != self.pad_value).bool().unsqueeze(0).expand(self.max_len, -1)

        out = PolarsDatasetOutput(id=id, x=x, y=y, mask=mask, weight=weight)
        return out
