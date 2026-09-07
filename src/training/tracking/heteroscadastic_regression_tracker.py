from logging import getLogger
from pathlib import Path
from typing import ClassVar, override

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (  # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]
    mean_absolute_error,
)
from utils.logging import setup_logger

from .metric_tracker import MetricTracker

logger = getLogger(__name__)
_ = setup_logger(logger)


def residuals_preds_plot(
    y_true,
    preds,
    sigmas,
    max_points_for_bars: int = 100,
):
    # Calculate Standardized (Normalized) Residuals
    residuals = y_true - preds
    standardized_residuals = residuals / sigmas

    # Set up the plotting environment
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- PLOT 1: Standardized Residuals vs. Targets ---
    sns.scatterplot(
        x=y_true,
        y=standardized_residuals,
        alpha=0.5,
        ax=axes[0],
        color="purple",
    )
    axes[0].axhline(0, color="black", linestyle="-", linewidth=1.5)
    axes[0].axhline(
        3, color="red", linestyle="--", label="+3 SD Boundary"
    )  # 3 sigma boundary
    axes[0].axhline(-3, color="red", linestyle="--", label="-3 SD Boundary")
    axes[0].set_xlabel("True Target Values")
    axes[0].set_ylabel("Standardized Residuals \n (Error / Sigma)")
    axes[0].set_title("Standardized Residuals vs. True Targets")
    axes[0].legend()

    # --- PLOT 2: Predictions with Uncertainty Bands ---
    # Sort data points by true value for cleaner line/ribbon plotting
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    preds_sorted = preds[sort_idx]
    sigmas_sorted = sigmas[sort_idx]

    if len(y_true) > max_points_for_bars:
        # Large dataset: Use continuous shaded ribbon
        sns.lineplot(
            x=y_true_sorted,
            y=preds_sorted,
            ax=axes[1],
            color="blue",
            label="Model Prediction",
        )
        axes[1].fill_between(
            y_true_sorted,
            preds_sorted - (3 * sigmas_sorted),
            preds_sorted + (3 * sigmas_sorted),
            color="blue",
            alpha=0.15,
            label=r"$\pm3\sigma$ Uncertainty Band",
        )
        # Overlay actual data points
        sns.scatterplot(
            x=y_true,
            y=y_true,
            color="red",
            alpha=0.3,
            s=10,
            ax=axes[1],
            label="Perfect Identity Line (True y)",
        )
    else:
        # Small dataset: Use discrete error bars
        axes[1].errorbar(
            x=y_true,
            y=preds,
            yerr=3 * sigmas,
            fmt="o",
            color="blue",
            ecolor="lightblue",
            elinewidth=2,
            capsize=3,
            alpha=0.7,
            label=r"Pred $\pm3\sigma$",
        )
        # Reference identity line
        sns.lineplot(
            x=y_true_sorted,
            y=y_true_sorted,
            color="red",
            linestyle="--",
            ax=axes[1],
            label="True Identity",
        )

    axes[1].set_xlabel("True Target Values")
    axes[1].set_ylabel("Predicted Values")
    axes[1].set_title(r"Model Predictions with $3\sigma$ Uncertainty")
    axes[1].legend()

    plt.tight_layout()
    return fig


def histplot(prefix: str, x: np.ndarray):
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(9, 5))

    sns.histplot(
        x,
        kde=True,
        color="#1f77b4",
        edgecolor="white",
        linewidth=1.2,
        alpha=0.6,
        bins="auto",
    )

    plt.title(
        f"Distribution of Values ({prefix})",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Value", fontsize=12, labelpad=10)
    plt.ylabel("Count / Density", fontsize=12, labelpad=10)
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    return fig


class HSRegregressionTracker(MetricTracker):
    """
    Heteroscedastic regression metrics tracker.

    Stores predictions, uncertainties (sigma), targets and ids, then computes
    MAE, WAPE, 3-sigma capture and diagnostic plots.
    """

    store_names: ClassVar[tuple[str, ...]] = (
        "train_ids",
        "train_preds",
        "train_sigma",
        "train_y",
        "train_y_orig",
        "train_loss",
        "val_ids",
        "val_preds",
        "val_sigma",
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
        preds_t: torch.Tensor,
        sigmas_t: torch.Tensor,
        y_true_t: torch.Tensor,
        step: int,
    ) -> None:
        """Plots normalized residuals and predictions with 3-sigma uncertainty bands."""
        ae = torch.abs(y_true_t - preds_t)
        ane = ae / sigmas_t
        ae = ae.detach().cpu().numpy().flatten()
        ane = ane.detach().cpu().numpy().flatten()

        y_true: np.ndarray = y_true_t.detach().cpu().numpy().flatten()
        preds: np.ndarray = preds_t.detach().cpu().numpy().flatten()
        sigmas: np.ndarray = sigmas_t.detach().cpu().numpy().flatten()
        del y_true_t, preds_t, sigmas_t

        fig1 = residuals_preds_plot(y_true=y_true, preds=preds, sigmas=sigmas)
        mlflow.log_figure(
            fig1,
            f"{prefix}-plots/sigma-vis/sigma-visual-{step}.png",
            save_kwargs={"dpi": 72},
        )
        fig2 = histplot(f"{prefix}-Absolute Error", ae)
        mlflow.log_figure(
            fig2,
            f"{prefix}-plots/residuals/residuals-hist-{step}.png",
            save_kwargs={"dpi": 72},
        )

        fig3 = histplot(f"{prefix}-Absolute Normalised Error", ane)
        mlflow.log_figure(
            fig3,
            f"{prefix}-plots/norm-residuals/norm-residuals-hist-{step}.png",
            save_kwargs={"dpi": 72},
        )

        fig4 = histplot(f"{prefix}-targets", y_true)
        mlflow.log_figure(
            fig4,
            f"{prefix}-plots/targets/targets-step-{step}.png",
            save_kwargs={"dpi": 72},
        )

    @override
    def calc_metrics(
        self,
        *,
        prefix: str,
        step: int,
    ) -> None:
        preds = self._gather_store(store_name=f"{prefix}_preds")
        sigmas = self._gather_store(store_name=f"{prefix}_sigma")
        y_true = self._gather_store(store_name=f"{prefix}_y")
        _ = self._gather_store(store_name=f"{prefix}_ids")

        self._log_plots(
            prefix, preds_t=preds, sigmas_t=sigmas, y_true_t=y_true, step=step
        )

        if preds.size(0) != y_true.size(0):
            logger.error(
                f"Different n. examples in preds and y_true: preds shape: {preds.shape}, y_true shape:{y_true.shape}"
            )
            return

        n_examples = preds.shape[0]

        try:
            sigma = torch.mean(sigmas).item()
            self.log_metric(f"{prefix}_sigma", sigma, n_examples)
        except Exception as e:
            logger.error(e)

        try:
            mae = mean_absolute_error(y_true.numpy(), preds.numpy())
            self.log_metric(f"{prefix}_MAE", mae, n_examples)
        except Exception as e:
            logger.error(e)

        try:
            mae = torch.abs(y_true - preds)
            wape = ((torch.sum(mae) / torch.sum(y_true)).item()) * 100
            self.log_metric(f"{prefix}_WAPE", wape, n_examples)
        except Exception as e:
            logger.error(e)

        try:
            upper = preds + (sigmas * 3)
            lower = preds - (sigmas * 3)
            within = (y_true <= upper) & (y_true >= lower)
            prop_within = torch.mean(within.float()).item()
            self.log_metric(f"{prefix}_3sd_capture", prop_within, n_examples)
        except Exception as e:
            logger.error(e)
