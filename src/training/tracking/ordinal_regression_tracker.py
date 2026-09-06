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
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from training.losses.wasserstein_funcs import wasserstein_loss
from utils.logging import setup_logger

from ..losses.entropy import norm_entropy_loss
from .metric_tracker import MetricTracker
from .plot_funcs import histplot

logger = getLogger(__name__)
_ = setup_logger(logger)


def ordinal_preds_plot(
    prefix: str,
    y_true_classes: np.ndarray,
    y_true_orig: np.ndarray,
    probs: np.ndarray,
    class_medians: np.ndarray,
    xlim=None,
    ylim=None,
):
    # Alternative/Better metric for center: Expected Value (Mean class)
    num_classes = probs.shape[-1]
    class_indices = np.arange(num_classes)
    expected_value = np.sum(probs * class_medians, axis=1)

    # The "Spread": Weighted standard deviation across the ordinal classes
    weighted_var = np.sum(
        probs * np.square(class_indices - expected_value[:, None]), axis=1
    )
    weighted_std = np.sqrt(weighted_var)

    # 3. Plotting
    fig = plt.figure(figsize=(10, 6))

    # Plot the background class lines for clean reference
    for i in range(num_classes):
        plt.axhline(y=i, color="gray", linestyle="--", alpha=0.3)

    # Plot the uncertainty band around the predictions
    # (Using expected_value here makes the ribbon smooth; you can swap to argmax_preds to see the steps)
    plt.fill_between(
        y_true_orig,
        expected_value - weighted_std,
        expected_value + weighted_std,
        color="purple",
        alpha=0.2,
        label=r"Prediction Spread ($\sigma$)",
    )

    # Plot the Argmax line (as requested)
    # Plot the Expected Value line (to show how it smooths out the transitions)
    plt.plot(
        y_true_orig,
        expected_value,
        color="purple",
        lw=2,
        linestyle=":",
        label="Expected Class Value (Continuous)",
    )
    plt.plot(
        y_true_orig,
        y_true_orig,
        color="red",
        lw=1,
        linestyle=":",
        label="Target Values (continuous",
    )

    # Clean up the Y-axis to show explicit class labels
    plt.yticks(class_indices, [f"Class {i}" for i in class_indices])
    plt.ylabel("Ordinal Output Classes")
    plt.xlabel("Target original value")
    plt.title("Ordinal Regression: Discrete Predictions vs. Probability Spread")
    plt.legend()
    plt.grid(axis="x", alpha=0.3)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(xlim)

    return fig


class OrdinalRegressionTracker(MetricTracker):
    """
    Ordinal regression metrics tracker.

    Stores logits, probabilities, ordinal targets, original targets and ids,
    then computes ordinal-aware classification metrics.
    """

    store_names: ClassVar[tuple[str, ...]] = (
        "train_ids",
        "train_logits",
        "train_probs",
        "train_y",
        "train_y_orig",
        "train_loss",
        "val_ids",
        "val_logits",
        "val_probs",
        "val_y",
        "val_y_orig",
        "val_loss",
    )

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        export: bool = False,
        export_loc: Path | None = None,
    ) -> None:
        super().__init__(
            device=device,
            dtype=dtype,
            export=export,
            export_loc=export_loc,
        )

    @override
    def _log_plots(
        self,
        prefix: str,
        y_true: np.ndarray,
        probs: np.ndarray,
        preds: np.ndarray,
        step: int,
    ) -> None:

        fig1 = histplot(f"{prefix}-targets", y_true)
        mlflow.log_figure(
            fig1,
            f"{prefix}-plots/targets/targets-step-{step}.png",
            save_kwargs={"dpi": 72},
        )

        fig4 = histplot(f"{prefix}-predictions", preds)
        mlflow.log_figure(
            fig4,
            f"{prefix}-plots/preds/preds-step-{step}.png",
            save_kwargs={"dpi": 72},
        )

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
        y_true_orig = self._gather_store(store_name=f"{prefix}_y_orig")
        _ = self._gather_store(store_name=f"{prefix}_ids")
        try:
            self._log_plots(
                prefix=prefix,
                y_true=y_true.long().numpy(),
                probs=probs.numpy(),
                preds=torch.argmax(probs, dim=-1).numpy(),
                step=step,
            )
        except Exception as e:
            logger.error(e)

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
            y_true_one_hot = F.one_hot(y_true, num_classes=probs.shape[-1]).squeeze(1)
            w_loss = wasserstein_loss(probs, y_true_one_hot)
            self.log_metric(f"{prefix}_WassersteinDist", w_loss, n_examples)
        except Exception as e:
            logger.error(e)

        try:
            entropy = norm_entropy_loss(probs)
            self.log_metric(f"{prefix}_entropy", entropy.item(), probs.shape[0])
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
            roc_auc_indiv = roc_auc_score(  # pyright: ignore[reportUnknownVariableType]
                y_true.squeeze(1).long().numpy(),
                probs.numpy(),
                multi_class="ovr",
                average=None,
            )

            for i, score in enumerate(roc_auc_indiv):
                self.log_metric(f"{prefix}_{i}_ROC_AUC", score, n_examples)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)

        try:
            av_prec_score = average_precision_score(
                y_true.squeeze(1).long().numpy(),
                probs.numpy(),
                average="weighted",
            )
            self.log_metric(f"{prefix}_w_PR_AUC", av_prec_score, n_examples)  # pyright: ignore[reportArgumentType]

            av_prec_score_macro = average_precision_score(
                y_true.squeeze(1).long().numpy(),
                probs.numpy(),
                average="macro",
            )
            self.log_metric(f"{prefix}_macro_PR_AUC", av_prec_score_macro, n_examples)  # pyright: ignore[reportArgumentType]

            av_prec_indiv = average_precision_score(  # pyright: ignore[reportUnknownVariableType]
                y_true.squeeze(1).long().numpy(),
                probs.numpy(),
                average=None,
            )

            for i, score in enumerate(av_prec_indiv):
                self.log_metric(f"{prefix}_{i}_PR_AUC", score, n_examples)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)
        preds = torch.argmax(
            probs,
            dim=1,
        ).squeeze(-1)

        try:
            f1 = f1_score(y_true, preds, average="weighted")
            self.log_metric(f"{prefix}_w_F1", f1, n_examples)  # pyright: ignore[reportArgumentType]

            f1_macro = f1_score(y_true, preds, average="macro")
            self.log_metric(f"{prefix}_macro_F1", f1_macro, n_examples)  # pyright: ignore[reportArgumentType]

            f1_indiv = f1_score(y_true, preds, average=None)
            for i, score in enumerate(f1_indiv):
                self.log_metric(f"{prefix}_{i}_F1", score, n_examples)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)
        try:
            balanced_accuracy = balanced_accuracy_score(y_true, preds)
            self.log_metric(f"{prefix}_w_accuracy", balanced_accuracy, n_examples)  # pyright: ignore[reportArgumentType]
            acc_score = accuracy_score(y_true, preds)
            self.log_metric(f"{prefix}_accuracy", acc_score, n_examples)  # pyright: ignore[reportArgumentType]

        except Exception as e:
            logger.error(e)
        try:
            recall = recall_score(
                y_true.long().numpy(),
                preds.numpy(),
                average="weighted",
            )
            self.log_metric(f"{prefix}_w_recall", recall, n_examples)
            recall_macro = recall_score(
                y_true.long().numpy(),
                preds.numpy(),
                average="macro",
            )
            self.log_metric(f"{prefix}_macro_recall", recall_macro, n_examples)

            recall_indiv = recall_score(
                y_true.long().numpy(),
                preds.numpy(),
                average=None,
            )
            for i, score in enumerate(recall_indiv):
                self.log_metric(f"{prefix}_{i}_recall", score, n_examples)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)
        try:
            precision = precision_score(
                y_true.long().numpy(),
                preds.numpy(),
                average="weighted",
            )
            self.log_metric(f"{prefix}_w_precision", precision, n_examples)

            precision_macro = precision_score(
                y_true.long().numpy(),
                preds.numpy(),
                average="macro",
            )
            self.log_metric(f"{prefix}_macro_precision", precision_macro, n_examples)

            precision_indiv = precision_score(
                y_true.long().numpy(),
                preds.numpy(),
                average=None,
            )
            for i, score in enumerate(precision_indiv):
                self.log_metric(f"{prefix}_{i}_precision", score, n_examples)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)
