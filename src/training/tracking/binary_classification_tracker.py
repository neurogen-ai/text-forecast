from logging import getLogger
from pathlib import Path
from typing import ClassVar, override

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (  # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor
from utils.logging import setup_logger

from .metric_tracker import MetricTracker
from .plot_funcs import histplot

logger = getLogger(__name__)
_ = setup_logger(logger)


class BinaryClassificationTracker(MetricTracker):
    """
    Binary classification metrics tracker.

    Stores logits, probabilities, targets and ids, then computes binary
    classification metrics and diagnostic plots.
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
        try:
            fig1 = histplot(f"{prefix}-targets", y_true)
            mlflow.log_figure(
                fig1,
                f"{prefix}-plots/targets/targets-step-{step}.png",
                save_kwargs={"dpi": 72},
            )
        except Exception as e:
            logger.error(e)
        try:
            fig4 = histplot(f"{prefix}-predictions", preds)
            mlflow.log_figure(
                fig4,
                f"{prefix}-plots/preds/preds-step-{step}.png",
                save_kwargs={"dpi": 72},
            )
        except Exception as e:
            logger.error(e)
        try:
            roc_plot = RocCurveDisplay.from_predictions(y_true, probs)
            mlflow.log_figure(
                roc_plot.figure_,
                f"{prefix}-plots/ROC/roc_curve-step-{step}.png",
                save_kwargs={"dpi": 72},
            )
        except Exception as e:
            logger.error(e)

        try:
            pr_plot = PrecisionRecallDisplay.from_predictions(  # pyright: ignore[reportUnknownMemberType]
                y_true,
                probs,
                plot_chance_level=True,
            )
            mlflow.log_figure(
                pr_plot.figure_,
                f"{prefix}-plots/PR/pr_curve-step-{step}.png",
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
        if logits.size(0) != y_true.size(0):
            logger.error(
                f"Different n. examples in logits and y_true: logits shape: {logits.shape}, y_true shape:{y_true.shape}"
            )
            return

        n_out = self._infer_n_out(logits)
        if n_out > 1:
            self._calc_multilabel_metrics(
                prefix=prefix, step=step, probs=probs, y_true=y_true, n_out=n_out
            )
            return

        n_examples = probs.shape[0]
        try:
            mae = mean_absolute_error(y_true, probs)
            self.log_metric(f"{prefix}_MAE", mae, n_examples)
        except Exception as e:
            logger.error(e)
        try:
            roc_auc = roc_auc_score(  # pyright: ignore[reportUnknownVariableType]
                y_true.long().numpy(),
                probs.numpy(),
                multi_class="ovo",
                average="weighted",
            )
            self.log_metric(f"{prefix}_roc_auc", roc_auc, n_examples)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)

        try:
            pr_auc = average_precision_score(y_true.long().numpy(), probs.numpy())
            self.log_metric(f"{prefix}_PR_AUC", pr_auc, n_examples)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            try:
                mssg = f"{e}\nrecall:{recall}\nprecision:{precision}"
            except:
                mssg = str(e)
            logger.error(mssg)

        # binary case
        preds = torch.zeros_like(probs)
        preds[probs > 0.5] = 1

        self._log_plots(
            prefix=prefix,
            y_true=y_true.long().squeeze(-1).numpy(),
            probs=probs.squeeze(-1).numpy(),
            preds=preds.squeeze(-1).numpy(),
            step=step,
        )

        try:
            balanced_accuracy = balanced_accuracy_score(y_true, preds)
            self.log_metric(
                f"{prefix}_balanced_accuracy:50", balanced_accuracy, n_examples
            )  # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)

        try:
            f1 = f1_score(y_true, preds)
            self.log_metric(f"{prefix}_F1:50", f1, n_examples)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)

        try:
            recall = recall_score(y_true.long().numpy(), preds.numpy())
            self.log_metric(f"{prefix}_recall:50", recall, n_examples)
        except Exception as e:
            logger.error(e)
        try:
            precision = precision_score(y_true.long().numpy(), preds.numpy())
            self.log_metric(f"{prefix}_precision:50", precision, n_examples)
        except Exception as e:
            logger.error(e)

        thetas = np.linspace(0.25, 0.75, num=20, endpoint=False)
        accuracy_scores = []
        theta = None

        try:
            for theta in thetas:
                preds = torch.zeros_like(probs)
                preds[probs > theta] = 1
                preds = preds.numpy()

                balanced_accuracy = balanced_accuracy_score(y_true, preds)

                accuracy_scores.append(balanced_accuracy)

            accuracy_scores = np.array(accuracy_scores)
            high_accuracy_i = np.argmax(accuracy_scores)

            preds = torch.zeros_like(probs)
            preds[probs > thetas[high_accuracy_i]] = 1
            preds = preds.numpy()

            recall = recall_score(y_true, preds)
            precision = precision_score(y_true, preds)

            self.log_metric(
                f"{prefix}_best_accuracy", accuracy_scores[high_accuracy_i], n_examples
            )  # pyright: ignore[reportArgumentType]
            self.log_metric(
                f"{prefix}_best_accuracy_theta", thetas[high_accuracy_i], n_examples
            )  # pyright: ignore[reportArgumentType]
            self.log_metric(f"{prefix}_precision:best_acc", precision, n_examples)  # pyright: ignore[reportArgumentType]
            self.log_metric(f"{prefix}_recall:best_acc", recall, n_examples)  # pyright: ignore[reportArgumentType]

        except Exception as e:
            logger.error(f"theta: {theta} | error: {e}")

    def _calc_multilabel_metrics(
        self,
        *,
        prefix: str,
        step: int,
        probs: Tensor,
        y_true: Tensor,
        n_out: int,
    ) -> None:
        """Multi-label path (sub-plan 1.4.5).

        Per-class threshold metrics (0.5) aggregated macro and micro, plus
        per-class PR/ROC plots capped at the first 10 classes to protect
        MLflow artifact volume.
        """
        if probs.size(0) != y_true.size(0) or y_true.numel() != probs.numel():
            logger.error(
                f"Different n. examples in logits and y_true: logits shape: {probs.shape}, y_true shape:{y_true.shape}"
            )
            return

        probs = probs.reshape(-1, n_out)
        y_multi = (y_true.reshape(-1, n_out) > 0.5).long()
        preds = (probs > 0.5).long()
        n_examples = probs.shape[0]

        for average in ("macro", "micro"):
            try:
                f1 = f1_score(
                    y_multi.numpy(), preds.numpy(), average=average, zero_division=0  # pyright: ignore[reportCallIssue, reportArgumentType]
                )
                self.log_metric(f"{prefix}_F1:{average}", f1, n_examples)
            except Exception as e:
                logger.error(e)
            try:
                precision = precision_score(
                    y_multi.numpy(), preds.numpy(), average=average, zero_division=0  # pyright: ignore[reportCallIssue, reportArgumentType]
                )
                self.log_metric(
                    f"{prefix}_precision:{average}", precision, n_examples
                )
            except Exception as e:
                logger.error(e)
            try:
                recall = recall_score(
                    y_multi.numpy(), preds.numpy(), average=average, zero_division=0  # pyright: ignore[reportCallIssue, reportArgumentType]
                )
                self.log_metric(f"{prefix}_recall:{average}", recall, n_examples)
            except Exception as e:
                logger.error(e)

        plot_cap = min(n_out, 10)
        if plot_cap < n_out:
            logger.info(
                f"{n_out} classes present, plotting first {plot_cap} only"
            )
        for c in range(plot_cap):
            try:
                roc_plot = RocCurveDisplay.from_predictions(  # pyright: ignore[reportUnknownMemberType]
                    y_multi[:, c].numpy(), probs[:, c].numpy()
                )
                mlflow.log_figure(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                    roc_plot.figure_,  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                    f"{prefix}-plots/ROC/roc_curve-class-{c}-step-{step}.png",
                    save_kwargs={"dpi": 72},
                )
            except Exception as e:
                logger.error(e)
            try:
                pr_plot = PrecisionRecallDisplay.from_predictions(  # pyright: ignore[reportUnknownMemberType]
                    y_multi[:, c].numpy(), probs[:, c].numpy()
                )
                mlflow.log_figure(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                    pr_plot.figure_,  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                    f"{prefix}-plots/PR/pr_curve-class-{c}-step-{step}.png",
                    save_kwargs={"dpi": 72},
                )
            except Exception as e:
                logger.error(e)
