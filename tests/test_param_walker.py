"""Unit tests for the scalar param walker (plans/implementation/v2.3.1.md,
v2.3.1-param-walker commit 1). collect_scalars keeps str/int/float/bool,
stringifies other objects as their class name with None as "None", and
raises KeyError on unknown differing-value bare-key collisions.
log_params_guarded truncates to the MLflow caps, no-ops identical values,
and warns while keeping the original on changed values. MLflow is mocked;
no server.
"""

import sys
import types
from dataclasses import dataclass, field

import pytest

from training.tracking.params import (
    MLFLOW_MAX_PARAM_KEY_LENGTH,
    MLFLOW_MAX_PARAM_VALUE_LENGTH,
    Leaf,
    collect_scalars,
    log_params_guarded,
)


class _LiveObject:
    """Stands in for StrategyConfig's model/tracker/stream/device fields."""


@dataclass
class _Inner:
    embed_dim: int = 8


@dataclass
class _Config:
    lr: float = 0.001
    epochs: int = 3
    flag: bool = True
    name: str = "run"
    sub: _Inner = field(default_factory=_Inner)
    live: _LiveObject = field(default_factory=_LiveObject)
    missing: str | None = None


def test_dataclass_fields_become_leaves_with_path_origins():
    leaves = collect_scalars(("strategy", _Config()))
    assert leaves["strategy.lr"] == Leaf(value=0.001, origin="strategy.lr")
    assert leaves["strategy.epochs"] == Leaf(value=3, origin="strategy.epochs")
    assert leaves["strategy.flag"] == Leaf(value=True, origin="strategy.flag")
    assert leaves["strategy.name"] == Leaf(value="run", origin="strategy.name")


def test_nested_dataclass_recurses_and_live_objects_stringify():
    leaves = collect_scalars(("strategy", _Config()))
    assert leaves["strategy.sub.embed_dim"] == Leaf(
        value=8, origin="strategy.sub.embed_dim"
    )
    assert leaves["strategy.live"] == Leaf(
        value="LiveObject", origin="strategy.live"
    )


def test_none_stringifies_as_none():
    leaves = collect_scalars(("strategy", _Config()))
    assert leaves["strategy.missing"] == Leaf(
        value="None", origin="strategy.missing"
    )


def test_mapping_root_is_explicit_leaves():
    leaves = collect_scalars(
        ("train", {"batch_size": 32, "examples": 100}),
        ("experiment", {"experiment_name": "exp", "model_class": "Foo"}),
    )
    assert leaves["train.batch_size"] == Leaf(value=32, origin="train.batch_size")
    assert leaves["train.examples"] == Leaf(value=100, origin="train.examples")
    assert leaves["experiment.experiment_name"] == Leaf(
        value="exp", origin="experiment.experiment_name"
    )


def test_same_value_bare_key_collision_collapses():
    leaves = collect_scalars(("a", {"x": 1}), ("b", {"x": 1}))
    assert len(leaves) == 2


def test_unknown_differing_value_collision_raises_key_error():
    with pytest.raises(KeyError, match="'a.x' and 'b.x'"):
        collect_scalars(("a", {"x": 1}), ("b", {"x": 2}))


def test_scalar_root_becomes_a_leaf():
    leaves = collect_scalars(("epochs", 5))
    assert leaves == {"epochs": Leaf(value=5, origin="epochs")}


class TestPydanticModels:
    def test_pydantic_model_fields_recurse(self):
        pytest.importorskip("pydantic")
        from pydantic import BaseModel

        class _Spec(BaseModel):
            lr: float = 0.01
            weight_decay: float = 0.1

        class _ModelCfg(BaseModel):
            embed_dim: int = 64
            dropout: float = 0.1
            spec: _Spec = _Spec()

        leaves = collect_scalars(("model.config", _ModelCfg()))
        assert leaves["model.config.embed_dim"].value == 64
        assert leaves["model.config.spec.lr"].value == 0.01
        assert leaves["model.config.spec.weight_decay"].value == 0.1


class _FakeRun:
    def __init__(self, params: dict[str, str]):
        self.info = types.SimpleNamespace(run_id="run-1")
        self.data = types.SimpleNamespace(params=params)


@pytest.fixture()
def mlflow_stub(monkeypatch):
    calls: dict[str, object] = {"logged": {}}
    stub = types.ModuleType("mlflow")
    stub.active_run = lambda: _FakeRun({"epochs": "3"})
    stub.get_run = lambda run_id: _FakeRun({"epochs": "3"})
    stub.log_params = lambda params: calls["logged"].update(params)
    monkeypatch.setitem(sys.modules, "mlflow", stub)
    yield stub, calls


class TestLogParamsGuarded:
    def test_logs_new_params(self, mlflow_stub):
        _, calls = mlflow_stub
        log_params_guarded({"lr": "0.001", "epochs": "3"})
        assert calls["logged"] == {"lr": "0.001"}

    def test_identical_value_is_a_no_op(self, mlflow_stub):
        _, calls = mlflow_stub
        log_params_guarded({"epochs": "3"})
        assert calls["logged"] == {}

    def test_changed_value_warns_and_keeps_original(self, mlflow_stub, caplog):
        _, calls = mlflow_stub
        with caplog.at_level("WARNING"):
            log_params_guarded({"epochs": "5"})
        assert calls["logged"] == {}
        assert any("immutable" in r.message for r in caplog.records)

    def test_truncates_long_values_and_keys(self, mlflow_stub):
        _, calls = mlflow_stub
        log_params_guarded(
            {"k" * 400: "v" * 10000}
        )
        logged_key = next(iter(calls["logged"]))
        assert len(logged_key) == MLFLOW_MAX_PARAM_KEY_LENGTH
        assert len(calls["logged"][logged_key]) == MLFLOW_MAX_PARAM_VALUE_LENGTH

    def test_requires_active_run(self, monkeypatch):
        stub = types.ModuleType("mlflow")
        stub.active_run = lambda: None
        monkeypatch.setitem(sys.modules, "mlflow", stub)
        with pytest.raises(RuntimeError, match="active MLflow run"):
            log_params_guarded({"a": "b"})


def test_tabled_differing_value_collision_is_allowed():
    # embed_dim is in SUFFIX_RULES; resolve_keys disambiguates it, so the
    # walker must not raise here.
    leaves = collect_scalars(
        ("model.config.embedder", {"embed_dim": 8}),
        ("model.config.forecaster", {"embed_dim": 16}),
    )
    assert len(leaves) == 2
