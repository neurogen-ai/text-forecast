from typing import Protocol, TypeVar, cast

import torch
from torch import Tensor

from data.datasets.types import TokenBatch

from .base import BaseStrategy, StrategyConfig


class _ForwardOutput(Protocol):
    """Minimal protocol for the model outputs this strategy consumes.

    Logits only (sub-plan 1.4.3): probabilities are derived here via a
    detached sigmoid when the tracker needs them.
    """

    logits: Tensor


T_Batch = TypeVar("T_Batch", bound=TokenBatch)


class ClassificationStrategy(BaseStrategy[TokenBatch]):
    """Binary classification strategy over :class:`TokenBatch`.

    Migrates the historical ``ClassificationTrainer._step`` and
    ``ClassificationEvaluator._step`` logic verbatim (F1/F3), with batch-level
    metric logging owned by the tracker.
    """

    def move_to_device(self, batch: TokenBatch) -> TokenBatch:
        """Move a NamedTuple batch to the strategy device via the CUDA stream."""
        batch_type = type(batch)
        with self.stream:
            moved = tuple(self._to_device(item, self.device) for item in batch)
        self.stream_sync()
        return cast(TokenBatch, batch_type(*moved))  # type: ignore[call-overload]

    def training_step(self, batch: TokenBatch) -> float:
        """One training step: forward, backward, optimizer step, logging."""
        self.model.train()

        self.tracker.process_values((batch.y.clone(),), ("train_y",))
        batch = self.move_to_device(batch)

        out = cast(_ForwardOutput, self.model.forward(batch))
        loss = self.loss_fn(out, batch)
        loss_cpu = loss.detach().clone()

        loss = loss / self.accumulation_steps
        loss.backward()

        if (
            self.batch_i + 1
        ) % self.accumulation_steps == 0 or self.examples_per_epoch - batch.x.shape[
            0
        ] == self._batch_steps_i:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        loss_cpu_item = loss_cpu.item()
        self.tracker.log_metric("train_loss", loss_cpu_item, batch.x.shape[0])
        self.tracker.process_values(
            (out.logits.detach().clone(), torch.sigmoid(out.logits.detach().clone())),
            ("train_logits", "train_probs"),
        )

        assert self.epoch_i is not None
        assert self._batch_steps_i is not None
        batch_step = int(
            ((self.epoch_i - 1) * self.examples_per_epoch) + self._batch_steps_i
        )
        self.tracker.log_batch_metric(
            "train_loss-batch", loss_cpu_item, step=batch_step
        )

        self._batch_i += 1
        self._batch_steps_i += batch.x.shape[0]
        self._total_steps_i += batch.x.shape[0]

        return loss_cpu_item

    def validation_step(self, batch: TokenBatch) -> float:
        """One validation step: no gradients, prefix stores are ``val_*``."""
        self.model.eval()

        self.tracker.process_values((batch.y.clone(),), ("val_y",))
        self.tracker.process_values((batch.id.clone(),), ("val_ids",))
        batch = self.move_to_device(batch)

        with torch.no_grad():
            out = cast(_ForwardOutput, self.model.forward(batch))
            loss = self.loss_fn(out, batch)
            loss_cpu = loss.detach().clone()

        loss_cpu_item = loss_cpu.item()
        self.tracker.log_metric("val_loss", loss_cpu_item, batch.x.shape[0])
        self.tracker.process_values(
            (torch.sigmoid(out.logits.detach().clone()), out.logits.detach().clone()),
            ("val_probs", "val_logits"),
        )

        return loss_cpu_item
