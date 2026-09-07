"""v2.3.1: tracking-package exception handlers log tracebacks with context.

Every bare ``logger.error(e)`` / ``logger.error(str(e))`` site in
``src/training/tracking/`` must be a ``logger.exception("<context string>")``
call, so swallowed failures carry a traceback and name their site. Two
guards:

1. a ``caplog`` test on one representative site (a raising MAE metric in
   ``RegregressionTracker.calc_metrics``) asserting ``exc_info`` is
   present;
2. an AST scan asserting no ``logger.error(<bare name>)`` call remains
   anywhere in the package, so future bare handlers fail this test.
"""

import ast
from pathlib import Path

import pytest
import torch
import training.strategies.base  # noqa: F401 — breaks circular import
import training.tracking.regression_tracker as regression_module
from training.tracking import RegregressionTracker

TRACKING_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "training" / "tracking"


def _make_tracker() -> RegregressionTracker:
    tracker = RegregressionTracker(device=torch.device("cpu"), dtype=torch.float32)
    # no MLflow server in unit tests
    regression_module.mlflow.log_figure = lambda *a, **k: None  # type: ignore[assignment]
    return tracker


def _feed(tracker: RegregressionTracker) -> None:
    preds = torch.rand(16, 1)
    y = torch.rand(16, 1)
    ids = torch.arange(16).float().unsqueeze(-1)
    tracker.process_values((ids,), ("train_ids",))
    tracker.process_values((preds,), ("train_preds",))
    tracker.process_values((y,), ("train_y",))


def test_raising_metric_logs_exception_with_traceback(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatchFixture
) -> None:
    """The MAE handler must log via logger.exception with live exc_info.

    Poisoning cannot go through process_values: its NaN rejection empties the
    store, so calc_metrics would die on the empty-store fallback before
    reaching the MAE try block. Raise inside the metric instead, which is
    what the handler exists to report.
    """
    tracker = _make_tracker()
    tracker._log_plots = lambda **k: None  # plots not under test here
    _feed(tracker)

    def _raise(*args: object, **kwargs: object) -> object:
        raise RuntimeError("poisoned metric")

    monkeypatch.setattr(regression_module, "mean_absolute_error", _raise)

    with caplog.at_level("ERROR", logger="training.tracking.regression_tracker"):
        tracker.calc_metrics(prefix="train", step=0)

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "MAE metric computation failed in calc_metrics" in m for m in messages
    ), f"expected contextual exception message, got: {messages}"
    assert any(r.exc_info is not None for r in caplog.records), (
        "logger.exception must attach exc_info; the bare-error behaviour "
        "(message only, no traceback) has regressed"
    )


def test_no_bare_logger_error_calls_remain() -> None:
    """AST scan: every logger.error call must take a non-bare argument.

    A bare handler is one whose sole argument is a bare name (``e``, a
    prebuilt message variable) or ``str(...)``; those lose the traceback.
    Contextual f-strings and literal messages are fine.
    """
    offenders: list[str] = []
    for path in sorted(TRACKING_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "error"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
            ):
                continue
            arg = node.args[0] if node.args else None
            if arg is None:
                offenders.append(f"{path.name}:{node.lineno} logger.error() with no argument")
            elif isinstance(arg, ast.Name):
                offenders.append(f"{path.name}:{node.lineno} logger.error({arg.id})")
            elif isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "str":
                offenders.append(f"{path.name}:{node.lineno} logger.error(str(...))")
    assert offenders == [], "bare logger.error handlers remain:\n" + "\n".join(offenders)
