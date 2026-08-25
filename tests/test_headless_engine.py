"""2.0 step 2: headless (no-TTY) engine and tracker reporting.

The Engine must run to completion with ``progress=None`` and the tracker's
``report`` must tolerate a ``None`` progress_bar (a rich-less container has no
console to render into).
"""

from __future__ import annotations

import pytest
import torch

import training.tracking.binary_classification_tracker as bct_module
from training.tracking import BinaryClassificationTracker


def _make_tracker(monkeypatch: pytest.MonkeyPatch) -> BinaryClassificationTracker:
    tracker = BinaryClassificationTracker(
        device=torch.device("cpu"), dtype=torch.float32
    )
    # no MLflow server in unit tests; monkeypatch restores after each test
    monkeypatch.setattr(
        bct_module.mlflow, "log_figure", lambda *a, **k: None
    )
    return tracker


def _feed_binary(tracker: BinaryClassificationTracker, seed: int = 42, B: int = 32):
    torch.manual_seed(seed)
    logits = torch.randn(B, 1)
    probs = torch.sigmoid(logits)
    y = (torch.rand(B, 1) > 0.5).float()
    ids = torch.arange(B).float().unsqueeze(-1)
    tracker.process_values((ids,), ("train_ids",))
    tracker.process_values((logits,), ("train_logits",))
    tracker.process_values((probs,), ("train_probs",))
    tracker.process_values((y,), ("train_y",))


def test_report_without_progress_bar_returns_metrics(
    monkeypatch: pytest.MonkeyPatch,
):
    """Headless report returns aggregated metrics without rendering rich."""
    tracker = _make_tracker(monkeypatch)
    _feed_binary(tracker)
    tracker.calc_metrics(prefix="train", step=0)

    metrics = tracker.report(progress_bar=None, epoch=0)

    assert isinstance(metrics, dict)
    assert metrics
    for name, score in metrics.items():
        assert isinstance(score, float)
