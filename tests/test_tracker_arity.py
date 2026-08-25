"""Sub-plan 1.4.5: tracker arity awareness.

Binary path (n_out == 1) must produce byte-identical metrics to the
pre-change implementation; multi-label path must produce per-class
threshold metrics aggregated macro/micro and cap per-class plots.
"""

import json
from pathlib import Path

import pytest
import torch
import training.strategies.base  # noqa: F401 — breaks circular import
import training.tracking.binary_classification_tracker as bct_module
from training.tracking import BinaryClassificationTracker

SYNTH_BASELINE = Path("/tmp/binary_synth_baseline.json")


def _make_tracker() -> BinaryClassificationTracker:
    tracker = BinaryClassificationTracker(device=torch.device("cpu"), dtype=torch.float32)
    # no MLflow server in unit tests
    bct_module.mlflow.log_figure = lambda *a, **k: None  # type: ignore[assignment]
    return tracker


def _feed_binary(tracker: BinaryClassificationTracker, seed: int = 42, B: int = 64):
    torch.manual_seed(seed)
    logits = torch.randn(B, 1)
    probs = torch.sigmoid(logits)
    y = (torch.rand(B, 1) > 0.5).float()
    ids = torch.arange(B).float().unsqueeze(-1)
    tracker.process_values((ids,), ("train_ids",))
    tracker.process_values((logits,), ("train_logits",))
    tracker.process_values((probs,), ("train_probs",))
    tracker.process_values((y,), ("train_y",))


def test_binary_path_byte_identical():
    """Regression: n_out == 1 metrics match the pre-1.4.5 implementation."""
    if not SYNTH_BASELINE.exists():
        pytest.skip("synthetic baseline not captured; run capture script")
    expected = json.loads(SYNTH_BASELINE.read_text())

    tracker = _make_tracker()
    tracker._log_plots = lambda **k: None  # plots need matplotlib artists; not under test
    _feed_binary(tracker)
    tracker.calc_metrics(prefix="train", step=0)

    actual = {k: v[0].score for k, v in tracker.metric_store.items()}
    assert actual == expected


def test_multilabel_metrics_macro_micro():
    tracker = _make_tracker()
    torch.manual_seed(7)
    B, C = 32, 3
    probs = torch.rand(B, C)
    y = (torch.rand(B, C) > 0.5).float()
    ids = torch.arange(B).float().unsqueeze(-1)

    from sklearn.metrics import f1_score, precision_score, recall_score

    preds = (probs > 0.5).long()
    expected = {
        f"train_F1:{avg}": f1_score(y.long().numpy(), preds.numpy(), average=avg, zero_division=0)
        for avg in ("macro", "micro")
    }
    expected.update(
        {
            f"train_precision:{avg}": precision_score(
                y.long().numpy(), preds.numpy(), average=avg, zero_division=0
            )
            for avg in ("macro", "micro")
        }
    )
    expected.update(
        {
            f"train_recall:{avg}": recall_score(
                y.long().numpy(), preds.numpy(), average=avg, zero_division=0
            )
            for avg in ("macro", "micro")
        }
    )

    tracker.process_values((ids,), ("train_ids",))
    tracker.process_values((probs,), ("train_logits",))
    tracker.process_values((probs,), ("train_probs",))
    tracker.process_values((y,), ("train_y",))
    tracker.calc_metrics(prefix="train", step=0)

    actual = {k: v[0].score for k, v in tracker.metric_store.items()}
    for key, value in expected.items():
        assert key in actual, f"missing metric {key}"
        assert actual[key] == pytest.approx(value)


def test_multilabel_plots_capped(monkeypatch: pytest.MonkeyPatch):
    tracker = _make_tracker()
    logged: list[str] = []
    monkeypatch.setattr(
        bct_module.mlflow,
        "log_figure",
        lambda fig, path, **k: logged.append(path),
    )
    B, C = 16, 13  # above the 10-class cap
    probs = torch.rand(B, C)
    y = (torch.rand(B, C) > 0.5).float()
    ids = torch.arange(B).float().unsqueeze(-1)

    tracker.process_values((ids,), ("val_ids",))
    tracker.process_values((probs,), ("val_logits",))
    tracker.process_values((probs,), ("val_probs",))
    tracker.process_values((y,), ("val_y",))
    tracker.calc_metrics(prefix="val", step=1)

    class_ids = {int(p.split("class-")[1].split("-")[0]) for p in logged}
    assert max(class_ids) == 9, "plots must be capped at the first 10 classes"
