from logging import getLogger
from pathlib import Path
from typing import ClassVar, override

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (  # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from utils.logging import setup_logger

from .metric_tracker import MetricTracker
from .plot_funcs import histplot

logger = getLogger(__name__)
_ = setup_logger(logger)


class ClassificationTracker(MetricTracker):
    """
    Multi-class / binary classification metrics tracker.

    ``n_out`` controls whether predictions are treated as binary (``n_out == 1``)
    or multi-class (``n_out > 1``).
    """

    store_names: ClassVar[tuple[str, ...]] = (
        "train_ids",
        "train_logits",
        "train_probs",
        "train_y",
        "train_y_orig",
        "train_loss",
        "train_preds",
        "train_sigma",
        "val_ids",
        "val_logits",
        "val_probs",
        "val_y",
        "val_y_orig",
        "val_loss",
        "val_preds",
        "val_sigma",
    )

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_out: int,
        export: bool = False,
        export_loc: Path | None = None,
    ) -> None:
        super().__init__(
            device=device,
            dtype=dtype,
            export=export,
            export_loc=export_loc,
        )
        self.n_out = n_out

    @override
    def _log_plots(
        self,
        prefix: str,
        y_true: np.ndarray,
        probs: np.ndarray,
        step: int,
    ) -> None:

        fig1 = histplot(f"{prefix}-targets", y_true)
        mlflow.log_figure(
            fig1,
            f"{prefix}-plots/targets/targets-step-{step}.png",
            save_kwargs={"dpi": 72},
        )
        try:
            roc_plot = RocCurveDisplay.from_predictions(y_true, probs)
            mlflow.log_figure(
                roc_plot.figure_,
                f"{prefix}-plots/ROC/roc_curve-step-{step}.png",
                save_kwargs={"dpi": 72},
            )
        except Exception as e:
            logger.error(e)

            # OutPut Probs histogram
        try:
            # 1. Calculate Proportions for the Legend
            total = len(y_true)
            pos_count = np.sum(y_true == 1)
            neg_count = np.sum(y_true == 0)

            pos_pct = (pos_count / total) * 100
            neg_pct = (neg_count / total) * 100

            # 2. Set the visual style
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(10, 6))

            # We pass the arrays directly here
            sns.histplot(
                x=probs.squeeze(),
                hue=y_true,
                multiple="stack",
                palette={0: "red", 1: "green"},
                bins=40,
                edgecolor="white",
                alpha=0.7,
                ax=ax,
            )

            # 4. Customizing the Legend with Proportions
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0], [0], color="green", lw=4, label=f"Positive (1): {pos_pct:.1f}%"
                ),
                Line2D(
                    [0], [0], color="red", lw=4, label=f"Negative (0): {neg_pct:.1f}%"
                ),
            ]

            ax.legend(
                handles=legend_elements, title="Label Distribution", loc="upper right"
            )

            # 5. Styling and Limits
            plt.xlim(0, 1)
            plt.xlabel("Classifier Output (Probability)", fontsize=11)
            plt.ylabel("Count", fontsize=11)
            plt.title(
                "Histogram of Output Probabilities with target composition",
                fontsize=13,
                pad=15,
            )
            plt.tight_layout()

            mlflow.log_figure(
                fig,
                f"{prefix}-plots/ProbHist/-histogram-step-{step}.png",
                save_kwargs={"dpi": 72},
            )
        except Exception as e:
            logger.error(str(e))

    @override
    def calc_metrics(
        self,
        *,
        prefix: str,
        step: int,
    ) -> None:
        logits = self._gather_store(store_name=f"{prefix}_logits")
        probs = self._gather_store(store_name=f"{prefix}_probs")
        y_true = self._gather_store(store_name=f"{prefix}_y")
        _ = self._gather_store(store_name=f"{prefix}_ids")
        self._log_plots(prefix, y_true.long().numpy(), probs.numpy(), step)
        if logits.size(0) != y_true.size(0):
            logger.error(
                f"Different n. examples in logits and y_true: logits shape: {logits.shape}, y_true shape:{y_true.shape}"
            )
            return

        n_examples = probs.shape[0]
        try:
            y_true_one_hot = F.one_hot(y_true, num_classes=probs.shape[-1]).squeeze(1)
            mae = mean_absolute_error(y_true_one_hot, probs.squeeze(-1))
            self.log_metric(f"{prefix}_MAE", mae, n_examples)
        except Exception as e:
            logger.error(e)
        try:
            roc_auc = roc_auc_score(  # pyright: ignore[reportUnknownVariableType]
                y_true.squeeze(1).long().numpy(),
                probs.numpy(),
                multi_class="ovo",
                average="weighted",
            )
            self.log_metric(f"{prefix}_roc_auc", roc_auc, n_examples)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)

        if self.n_out == 1:
            # binary case
            preds = torch.zeros_like(probs)
            preds[probs > 0.5] = 1
        else:
            preds = torch.argmax(
                probs,
                dim=1,
            ).squeeze(-1)

        try:
            balanced_accuracy = balanced_accuracy_score(y_true, preds)
            self.log_metric(
                f"{prefix}_balanced_accuracy", balanced_accuracy, n_examples
            )  # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)
        try:
            recall = recall_score(
                y_true.long().numpy(),
                preds.numpy(),
                average="weighted",
            )
            self.log_metric(f"{prefix}_recall", recall, n_examples)
        except Exception as e:
            logger.error(e)
        try:
            precision = precision_score(
                y_true.long().numpy(),
                preds.numpy(),
                average="weighted",
            )
            self.log_metric(f"{prefix}_precision", precision, n_examples)
        except Exception as e:
            logger.error(e)
