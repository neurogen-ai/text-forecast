"""Id-feed tests for ClassificationStrategy (plan v2.3.1-id-feed).

Drives a minimal strategy with a recording tracker stub: real ids land in
``train_ids``, an all-NaN fallback batch skips the feed, and the val feed
behaves the same. Both feeds share ``_feed_ids``, so the guard
(``torch.isnan(batch.id).all()``) is exercised on both paths.
"""

from contextlib import nullcontext

import torch

from data.datasets.types import TokenBatch
from strategies.classification import ClassificationStrategy


class RecordingTracker:
    """Records process_values feeds as (values, store_names) tuples."""

    def __init__(self) -> None:
        self.feeds: list[tuple[tuple[torch.Tensor, ...], tuple[str, ...]]] = []

    def process_values(
        self, values: tuple[torch.Tensor, ...], store_names: tuple[str, ...]
    ) -> None:
        self.feeds.append((values, store_names))

    def feed_names(self) -> list[str]:
        return [names[0] for _, names in self.feeds]

    def log_metric(self, name: str, value: float, weight: int) -> None:  # noqa: ARG002
        return None

    def log_batch_metric(self, name: str, value: float, step: int) -> None:  # noqa: ARG002
        return None


class StubModel:
    """Minimal model stand-in: no parameters, fixed logits."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits

    def train(self) -> "StubModel":
        return self

    def eval(self) -> "StubModel":
        return self

    def forward(self, batch: TokenBatch) -> object:  # noqa: ARG002
        out = type("Out", (), {})()
        out.logits = self.logits
        return out


class StubLoss:
    def __call__(self, out: object, batch: TokenBatch) -> torch.Tensor:  # noqa: ARG002
        return torch.tensor(0.5)


def _make_strategy() -> tuple[ClassificationStrategy, RecordingTracker]:
    """Build a ClassificationStrategy without the pydantic/optimizer stack.

    Only the attributes the step paths touch are set; the optimizer is never
    stepped because accumulation_steps=2 and each test runs one batch.
    """
    tracker = RecordingTracker()
    strategy = object.__new__(ClassificationStrategy)
    strategy.model = StubModel(torch.zeros(2, 1))
    strategy.loss_fn = StubLoss()
    strategy.tracker = tracker
    strategy.device = torch.device("cpu")
    strategy.stream = nullcontext()
    strategy.stream_sync = lambda: None
    strategy.accumulation_steps = 2
    strategy.examples_per_epoch = 4
    strategy.epoch_i = 1
    strategy._batch_i = 0
    strategy._batch_steps_i = 0
    strategy._total_steps_i = 0
    return strategy, tracker


def _make_batch(id_values: list[float]) -> TokenBatch:
    return TokenBatch(
        id=torch.tensor(id_values),
        x=torch.zeros(2, 3, dtype=torch.long),
        y=torch.zeros(2, 1),
        mask=torch.ones(2, 3, dtype=torch.bool),
        weight=None,
        date=None,
    )


def test_training_step_feeds_real_train_ids() -> None:
    strategy, tracker = _make_strategy()
    strategy.training_step(_make_batch([1.0, 2.0]))
    assert "train_ids" in tracker.feed_names()
    feed = next(f for f in tracker.feeds if f[1] == ("train_ids",))
    assert torch.equal(feed[0][0], torch.tensor([1.0, 2.0]))


def test_training_step_skips_all_nan_fallback_batch() -> None:
    strategy, tracker = _make_strategy()
    strategy.training_step(_make_batch([float("nan"), float("nan")]))
    assert "train_ids" not in tracker.feed_names()
    # The non-id feeds still happen: the batch is processed, only ids skip.
    assert "train_y" in tracker.feed_names()
    assert "train_logits" in tracker.feed_names()


def test_validation_step_feeds_real_val_ids() -> None:
    strategy, tracker = _make_strategy()
    strategy.validation_step(_make_batch([7.0, 8.0]))
    feed = next(f for f in tracker.feeds if f[1] == ("val_ids",))
    assert torch.equal(feed[0][0], torch.tensor([7.0, 8.0]))


def test_validation_step_skips_all_nan_fallback_batch() -> None:
    strategy, tracker = _make_strategy()
    strategy.validation_step(_make_batch([float("nan"), float("nan")]))
    assert "val_ids" not in tracker.feed_names()
    assert "val_y" in tracker.feed_names()


def test_partially_nan_batch_still_feeds() -> None:
    """The guard skips only fully-NaN batches; a mixed batch is a real value."""
    strategy, tracker = _make_strategy()
    strategy.training_step(_make_batch([1.0, float("nan")]))
    assert "train_ids" in tracker.feed_names()
