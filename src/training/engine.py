"""Training/validation engine.

The :class:`Engine` owns the epoch and batch loops so that the CLI apps only
need to construct an :class:`Experiment` and call ``fit``.  This makes a later
cloud-execution push an Engine implementation swap rather than a rewrite of the
apps.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import mlflow
from rich.progress import Progress
from torch import Tensor
from torch.utils.data import DataLoader

from config.experiment import Experiment
from config.runtime import RunContext
from training.checkpointing import Checkpoint
from training.tracking.log_lrs import log_lrs


class _HasBatchSize(Protocol):
    """Minimal batch shape used by the engine to advance progress bars."""

    x: Tensor


T_Batch = TypeVar("T_Batch", bound=_HasBatchSize)
ProgressBars = tuple[Progress, Progress, Progress] | None


class Engine[T_Batch: _HasBatchSize]:
    """Epoch/batch loop driver for an :class:`Experiment`."""

    def __init__(
        self,
        *,
        experiment: Experiment[T_Batch],
        runtime: RunContext,
        progress: ProgressBars = None,
    ) -> None:
        self.experiment = experiment
        self.runtime = runtime
        self._progress = progress

        if progress is not None:
            self._epoch_task = progress[0].add_task(
                "Epochs", total=experiment.epochs
            )
            self._train_task = progress[1].add_task(
                "Train Examples",
                total=self._total_examples(experiment.train_loader),
            )
            self._val_task = progress[2].add_task(
                "Eval Examples",
                total=self._total_examples(experiment.val_loader),
            )

    def _advance(self, bar_idx: int, *, advance: int) -> None:
        """Advance one progress task when bars are active (headless-safe)."""
        if self._progress is None:
            return
        task = (self._train_task, self._val_task)[bar_idx]
        self._progress[bar_idx + 1].advance(task, advance=advance)

    def _reset(self, bar_idx: int, *, description: str, total: int) -> None:
        """Reset one progress task when bars are active (headless-safe)."""
        if self._progress is None:
            return
        task = (self._train_task, self._val_task)[bar_idx]
        bar = self._progress[bar_idx + 1]
        bar.reset(task, description=description, total=total)

    def _advance_epoch(self) -> None:
        """Advance the epoch task when bars are active (headless-safe)."""
        if self._progress is None:
            return
        self._progress[0].advance(self._epoch_task, advance=1)

    @staticmethod
    def _total_examples(loader: DataLoader[T_Batch]) -> int:
        """Derive progress totals from the sampler or dataset (J7)."""
        if loader.sampler is not None:
            return len(loader.sampler)
        return len(loader.dataset)

    @staticmethod
    def _batch_size(batch: T_Batch) -> int:
        return int(batch.x.shape[0])

    def train_epoch(self, epoch: int) -> None:
        """Run one training epoch and advance the train progress bar."""
        strategy = self.experiment.strategy
        strategy.start_epoch(epoch=epoch)

        self._reset(
            0,
            description="Train Examples",
            total=self._total_examples(self.experiment.train_loader),
        )

        for batch in self.experiment.train_loader:
            _ = strategy.training_step(batch)
            self._advance(0, advance=self._batch_size(batch))

    def eval_epoch(
        self,
        epoch: int,
        *,
        loader: DataLoader[T_Batch] | None = None,
    ) -> None:
        """Run one evaluation epoch and compute ``val_*`` metrics."""
        loader = loader or self.experiment.val_loader
        strategy = self.experiment.strategy

        self._reset(
            1,
            description="Eval Examples",
            total=self._total_examples(loader),
        )

        for batch in loader:
            _ = strategy.validation_step(batch)
            self._advance(1, advance=self._batch_size(batch))

        self.experiment.tracker.calc_metrics(prefix="val", step=epoch)

    def fit(self, *, start_epoch: int, run_id: str) -> None:
        """Run the full training schedule.

        The loop mirrors the historical ``apps/train.py`` flow lifted here in
        phase 5: train -> scheduler step -> checkpoint -> eval -> metric report.
        """
        strategy = self.experiment.strategy

        for epoch in range(start_epoch, start_epoch + self.experiment.epochs):
            log_lrs(strategy.scheduler, epoch)
            self.train_epoch(epoch)
            strategy.scheduler_step()

            if epoch % self.experiment.checkpoint_interval == 0:
                state = Checkpoint(
                    model=self.experiment.model.state_dict(),
                    optimizer=strategy.optimizer.state_dict(),
                    scheduler=strategy.scheduler.state_dict(),
                    epoch=epoch,
                )
                _ = self.experiment.checkpoints.save(
                    state=state, run_id=run_id
                )

            if epoch % self.experiment.eval_interval == 0:
                self.eval_epoch(epoch)

            self.experiment.tracker.calc_metrics(prefix="train", step=epoch)
            metrics = self.experiment.tracker.report(
                progress_bar=(
                    self._progress[0] if self._progress is not None else None
                ),
                epoch=epoch,
            )
            if metrics:
                mlflow.log_metrics(
                    metrics, step=epoch, synchronous=False
                )
            self.experiment.tracker.clear()
            self._advance_epoch()
