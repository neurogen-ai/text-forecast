"""Tests for the latched empty-store warning in MetricTracker._gather_store.

The empty-store message must log at WARNING the first time a store name comes
up empty and at DEBUG afterwards, once per store name per tracker instance,
surviving clear() (plan v2.3.1-store-latch, AD3).
"""

import logging

import pytest
import torch

from training.tracking.metric_tracker import MetricTracker


class _TwoStoreTracker(MetricTracker):
    store_names = ("train_ids", "val_ids")


@pytest.fixture()
def tracker():
    return _TwoStoreTracker(device=torch.device("cpu"), dtype=torch.float32)


TRACKER_LOGGER_NAME = logging.getLogger(MetricTracker.__module__).name


def _records_for(caplog, store_name):
    return [
        r for r in caplog.records
        if r.name == TRACKER_LOGGER_NAME and store_name in r.getMessage()
    ]


def test_first_empty_gather_warns_once(tracker, caplog):
    caplog.set_level(logging.DEBUG)
    with caplog.at_level(logging.DEBUG):
        out = tracker._gather_store(store_name="train_ids")
    assert out.item() != out.item()  # NaN returned, contract unchanged
    records = _records_for(caplog, "train_ids")
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING


def test_second_empty_gather_logs_debug(tracker, caplog):
    caplog.set_level(logging.DEBUG)
    tracker._gather_store(store_name="train_ids")
    caplog.clear()
    caplog.clear()
    tracker._gather_store(store_name="train_ids")
    records = _records_for(caplog, "train_ids")
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG


def test_fed_store_does_not_warn(tracker, caplog):
    caplog.set_level(logging.DEBUG)
    tracker.stores["train_ids"].append(torch.tensor([1.0, 2.0]))
    out = tracker._gather_store(store_name="train_ids")
    assert out.tolist() == [1.0, 2.0]
    assert _records_for(caplog, "train_ids") == []


def test_latch_survives_clear(tracker, caplog):
    caplog.set_level(logging.DEBUG)
    tracker._gather_store(store_name="train_ids")
    tracker.clear()
    caplog.clear()
    tracker._gather_store(store_name="train_ids")
    records = _records_for(caplog, "train_ids")
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG


def test_instances_latch_independently(tracker, caplog):
    caplog.set_level(logging.DEBUG)
    tracker._gather_store(store_name="val_ids")
    other = _TwoStoreTracker(device=torch.device("cpu"), dtype=torch.float32)
    caplog.clear()  # drop the first instance's records; only the new instance's count
    other._gather_store(store_name="val_ids")
    records = _records_for(caplog, "val_ids")
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
