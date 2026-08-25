import math
import os
from logging import getLogger
from pathlib import Path
from typing import ClassVar, NamedTuple

import mlflow
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (  # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from torch import Tensor

from utils.logging import setup_logger

from ..losses.entropy import norm_entropy_loss

logger = getLogger(__name__)
_ = setup_logger(logger)


class MetricTuple(NamedTuple):
    "Metric Tuple: stores named metric scores / weight values for dataframe concatenations"

    score: float
    weight: float


class MetricTracker:
    """
    Base metric tracker.

    Stores intermediate outputs as CPU-side tensors, calculates results, and
    displays them. Subclasses declare the store names they consume via
    ``store_names`` and override ``calc_metrics`` for task-specific math.

    args:
        device: torch device used for the run (kept for interface compatibility).
        dtype: torch dtype used for the run (kept for interface compatibility).
        export: whether to export gathered stores to disk.
        export_loc: directory to export stores to when ``export`` is True.
    """

    store_names: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        export: bool = False,
        export_loc: Path | None = None,
    ) -> None:
        assert export or export_loc is None, "Provide a location to export metrics to"

        self.dtype: torch.dtype = dtype
        self.device: torch.device = device
        self.export = export
        self.export_loc = export_loc
        self.stores: dict[str, list[Tensor]] = {
            name: [] for name in self.store_names
        }

        self._init_metric_stores()

        # rich console
        self.console: Console = Console()

    @staticmethod
    def _infer_n_out(logits: Tensor) -> int:
        """Infer output arity from a gathered logits tensor.

        The last dimension is the class count for ``(B, n_out)`` stores; flat
        or scalar-shaped stores count as single-output (binary).
        """
        return int(logits.shape[-1]) if logits.ndim > 1 else 1

    def _init_metric_stores(self) -> None:
        self.metric_store: dict[str, list[MetricTuple]] = {}
        self.metric_df: pd.DataFrame = pd.DataFrame()

    def _ensure_store(self, name: str) -> list[Tensor]:
        if name not in self.stores:
            self.stores[name] = []
        return self.stores[name]

    def clear(self) -> None:
        for name in list(self.stores.keys()):
            self.stores[name] = []
        self._init_metric_stores()

    def _process_value(self, value: Tensor, store_name: str) -> None:
        """Stores a single value in its CPU-side list after NaN-checking."""
        if torch.any(torch.isnan(value)):
            logger.error(
                f"NaN values passed to process_value, not writing to store {store_name}"
            )
            return

        store = self._ensure_store(store_name)
        store.append(value.detach().cpu())

    def process_values(
        self,
        values: tuple[Tensor, ...],
        store_names: tuple[str, ...],
    ) -> None:
        for i, value in enumerate(values):
            if torch.any(torch.isnan(value)):
                logger.error(
                    f"{store_names[i]} contains NaN values, {store_names} skipped"
                )
                return

        for i, value in enumerate(values):
            self._process_value(value, store_names[i])

    def _gather_store(
        self,
        *,
        store_name: str,
    ) -> Tensor:
        """Concatenates and flattens all stored buffers for a single store.

        out shape: (batch, *output_shape)
        """
        store = self._ensure_store(store_name)

        if len(store) > 0:
            all_values: Tensor = torch.cat(store)
            if self.export and self.export_loc is not None:
                logger.debug(
                    f"Exporting {store_name} store to {self.export_loc.resolve()}"
                )
                os.makedirs(self.export_loc, exist_ok=True)
                np.save(self.export_loc / store_name, all_values.numpy())
            self.stores[store_name] = []
            return all_values
        else:
            logger.error(f"Store {store_name} contains no values, resetting")
            self.stores[store_name] = []
            return torch.tensor(float("nan"))

    def log_metric(
        self,
        name: str,
        value: float,
        weight: float | int,
    ) -> None:
        if name not in self.metric_store.keys():
            self.metric_store[name] = []
        self.metric_store[name].append(MetricTuple(value, weight))

    def log_batch_metric(
        self,
        name: str,
        value: float,
        *,
        step: int,
    ) -> None:
        """Owns all batch-level metric logging, including ``train_loss-batch`` (F3)."""
        mlflow.log_metric(name, value, step=step, synchronous=False)

    def _log_plots(
        self,
        prefix: str,
        y_true: Tensor,
        probs: Tensor,
        step: int,
    ) -> None:
        try:
            roc_plot = RocCurveDisplay.from_predictions(
                y_true.long().numpy(), probs.numpy()
            )
            mlflow.log_figure(
                roc_plot.figure_,
                f"{prefix}-plots/ROC/roc_curve-step-{step}.png",
                save_kwargs={"dpi": 72},
            )
        except Exception as e:
            logger.error(e)

        try:
            pr_plot = PrecisionRecallDisplay.from_predictions(
                y_true.long().numpy(),
                probs.numpy(),
                plot_chance_level=True,
            )
            mlflow.log_figure(
                pr_plot.figure_,
                f"{prefix}-plots/PR/pr_curve-step-{step}.png",
                save_kwargs={"dpi": 72},
            )
        except Exception as e:
            logger.error(e)

    def calc_metrics(
        self,
        *,
        prefix: str,
        step: int,
    ) -> None:
        try:
            logits = self._gather_store(store_name=f"{prefix}_logits")
            probs = torch.softmax(logits, dim=-1)

            preds = torch.argmax(
                probs,
                dim=-1,
            ).unsqueeze(-1)
            y_true = self._gather_store(store_name=f"{prefix}_y")

        except Exception as e:
            logger.error(e)
            return

        if preds.size(0) != y_true.size(0):
            logger.error(
                f"Different n. examples in logits and y_true: logits shape: {probs.shape}, y_true shape:{y_true.shape}"
            )
            return

        try:
            entropy = norm_entropy_loss(probs)
            self.log_metric(f"{prefix}_entropy", entropy.item(), preds.shape[0])
        except Exception as e:
            logger.error(e)

    def _aggregate_metrics(self) -> dict[str, float]:
        "Aggregates metrics stored as named tuples"
        aggregate_metrics = {k: float("nan") for k in self.metric_store.keys()}
        for metric in aggregate_metrics.keys():
            try:
                df: pd.DataFrame = pd.DataFrame(self.metric_store[metric])
                score = (df["score"] * df["weight"]).sum() / (df["weight"].sum())
                if not math.isnan(score):
                    aggregate_metrics[metric] = round(score, 6)
            except Exception as e:
                logger.error(e)
        aggregate_metrics = {
            k: v for k, v in aggregate_metrics.items() if not math.isnan(v)
        }
        return aggregate_metrics

    def report(
        self,
        progress_bar: object | None = None,
        *,
        epoch: int | None = None,
    ) -> dict[str, float]:
        "Aggregate metrics; render a rich.Table when a progress_bar is given."
        aggregate_metrics = self._aggregate_metrics()

        if progress_bar is not None:
            cols: list[str] = ["cyan", "green"]
            table: Table = Table(
                show_header=True, pad_edge=True, padding=(0, 1)
            )
            table.add_column("Epoch", style="cyan")
            for i, (k) in enumerate(aggregate_metrics.keys()):
                table.add_column(k, style=cols[i % len(cols)])

            row_data = [str(epoch) if epoch is not None else "NA"]
            row_data.extend(str(v) for v in aggregate_metrics.values())
            table.add_row(*row_data)

            progress_bar.console.print(table)

        return aggregate_metrics
