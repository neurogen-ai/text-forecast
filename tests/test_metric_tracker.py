"""v2.4.1: bf16 metric stores crash numpy.

A bf16 store must round-trip ``_gather_store`` to numpy without raising
(numpy has no bfloat16), and the export path must write an fp32 .npy.
Requires torch; runs on the owner's machine or CI (this container has no
torch install).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import training.strategies.base  # noqa: F401 — breaks circular import
from training.tracking import MetricTracker


def _make_tracker(export_loc: Path | None = None) -> MetricTracker:
    tracker = MetricTracker(
        device=torch.device("cpu"),
        dtype=torch.float32,
        export=export_loc is not None,
        export_loc=export_loc,
    )
    return tracker


def test_bf16_store_round_trips_numpy():
    """A bf16 store converts to fp32 at the gather boundary (needs torch)."""
    if not hasattr(torch, "bfloat16"):
        pytest.skip("torch build lacks bfloat16")
    tracker = _make_tracker()
    # NOTE: needs torch — bf16 tensors cannot be produced without it.
    bf16_values = torch.randn(8, 1, dtype=torch.bfloat16)
    tracker.process_values((bf16_values,), ("train_probs",))

    gathered = tracker._gather_store(store_name="train_probs")

    # no raise on the numpy boundary
    np_values = gathered.numpy()
    assert np_values.dtype == np.float32
    assert np.allclose(np_values, bf16_values.float().numpy())


def test_bf16_export_writes_fp32_npy(tmp_path: Path):
    """The export path writes an fp32 .npy for a bf16 store (needs torch)."""
    if not hasattr(torch, "bfloat16"):
        pytest.skip("torch build lacks bfloat16")
    tracker = _make_tracker(export_loc=tmp_path)
    # NOTE: needs torch — bf16 tensors cannot be produced without it.
    bf16_values = torch.randn(8, 1, dtype=torch.bfloat16)
    tracker.process_values((bf16_values,), ("train_probs",))

    tracker._gather_store(store_name="train_probs")

    exported = np.load(tmp_path / "train_probs.npy")
    assert exported.dtype == np.float32


def test_fp32_store_passes_through_unchanged():
    """AD2: fp32 stores pass through without conversion."""
    tracker = _make_tracker()
    fp32_values = torch.randn(8, 1, dtype=torch.float32)
    tracker.process_values((fp32_values,), ("train_probs",))

    gathered = tracker._gather_store(store_name="train_probs")

    assert gathered.dtype == torch.float32
