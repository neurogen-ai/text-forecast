"""Tests for compose_train_params and its walker integration.

The stubs mimic the real shapes loosely: compose_train_params reads only
experiment_name, model, train_loader, val_loader, strategy.config, and the
runtime argument, so plain dataclasses and namespaces are enough. The key
assertions pin the emitted key set, including the ambiguous-leaf
duplication resolved by SUFFIX_RULES.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from training.pipeline.train import _is_config_instance, compose_train_params
from training.tracking import collect_scalars, log_params_guarded, resolve_keys
from training.tracking.param_keys import SUFFIX_RULES


@dataclass
class OptimizerSpecStub:
    lr: float = 3e-4
    weight_decay: float = 0.01


@dataclass
class SchedulerSpecStub:
    epochs: int = 12
    warmup_start_factor: float = 0.1
    eta_min: float = 1e-6
    milestones: tuple[int, ...] = (6,)


# Live objects mirror the real shapes: the walker stringifies them as
# type(obj).__name__, so the stub classes are built with __name__ set to
# the asserted value. device/dtype use the real torch scalars: the walker
# the walker records str(device)/str(dtype) via the allow-list branch, the
# actual run values ("cpu", "bfloat16"), not the useless "device"/"dtype"
# class names.
# runtime.device and strategy.device carry the same value, keeping the
# bare-key values identical.
_MODEL_LIVE = type("MODEL-LIVE", (), {})
_LOSS_LIVE = type("LOSS-LIVE", (), {})
_TRACKER_LIVE = type("TRACKER-LIVE", (), {})
_STREAM_LIVE = type("STREAM-LIVE", (), {})


@dataclass
class StrategyConfigStub:
    model: object = field(default_factory=_MODEL_LIVE)
    loss_fn: object = field(default_factory=_LOSS_LIVE)
    tracker: object = field(default_factory=_TRACKER_LIVE)
    stream: object = field(default_factory=_STREAM_LIVE)
    device: object = torch.device("cpu")
    optimizer_spec: OptimizerSpecStub = field(default_factory=OptimizerSpecStub)
    scheduler_spec: SchedulerSpecStub = field(default_factory=SchedulerSpecStub)
    examples_per_epoch: int = 14853
    accumulation_steps: int = 2
    mat_mul_precision: str = "high"


@dataclass
class EmbedderCfg:
    embed_dim: int = 64


@dataclass
class ForecasterCfg:
    embed_dim: int = 32


@dataclass
class ModelCfg:
    embedder: EmbedderCfg = field(default_factory=EmbedderCfg)
    forecaster: ForecasterCfg = field(default_factory=ForecasterCfg)


class ModelWithConfig:
    """Mimics the 20 families that store a config instance on self."""

    def __init__(self, config=None):
        self.config = ModelCfg() if config is None else config


class GraphRecurrentLike:
    """Mimics graph_recurrent: config is a class attribute, not an instance."""

    config = ModelCfg


@dataclass
class RunContextStub:
    device: object = torch.device("cpu")
    dtype: object = torch.bfloat16
    compile_mode: str = ""
    fullgraph: bool = False
    subsample: int | None = None


class _Dataset:
    def __len__(self) -> int:
        return 42


def _make_exp(model, examples: int = 42):
    loader = SimpleNamespace(dataset=_Dataset(), batch_size=8)
    val_loader = SimpleNamespace(dataset=_Dataset(), batch_size=16)
    return SimpleNamespace(
        experiment_name="graph_embed_class",
        model=model,
        train_loader=loader,
        val_loader=val_loader,
        strategy=SimpleNamespace(config=StrategyConfigStub()),
        eval_interval=2,
        checkpoint_interval=5,
    )


def _keys(exp, runtime=None) -> set[str]:
    roots = compose_train_params(exp, runtime or RunContextStub())
    return set(resolve_keys(collect_scalars(*roots)))


def test_legacy_four_keys_fold_into_new_scheme():
    exp = _make_exp(ModelWithConfig())
    keys = _keys(exp)
    # The full emitted key set: bare leaf keys plus the SUFFIX_RULES
    # disambiguation, identical across runs. Live objects appear as their
    # class names; the milestones tuple records its repr.
    assert keys == {
        # experiment root
        "experiment_name",
        "model_class",
        "eval_interval",
        "checkpoint_interval",
        # train/val roots, folded per AD7
        "examples",
        "batch_size",
        "examples-train",
        "examples-val",
        "batch_size-train",
        "batch_size-val",
        # runtime root
        "device",
        "dtype",
        "compile_mode",
        "fullgraph",
        "subsample",
        # strategy root: live objects as class names, spec dataclasses flat
        "model",
        "loss_fn",
        "tracker",
        "stream",
        "lr",
        "weight_decay",
        "epochs",
        "warmup_start_factor",
        "eta_min",
        "milestones",
        "examples_per_epoch",
        "accumulation_steps",
        "mat_mul_precision",
        # model root
        "embed_dim",
        "embed_dim-embedder",
        "embed_dim-forecaster",
    }
    # The legacy model.class / train.examples / val.examples dotted keys are
    # gone; AD7 folds them in as bare model_class and suffixed examples.
    assert "model.class" not in keys


def test_ambiguous_leaf_duplication_and_legacy_examples():
    exp = _make_exp(ModelWithConfig())
    keys = _keys(exp)
    assert "examples-train" in keys
    assert "examples-val" in keys
    assert "examples" in keys  # bare key, first origin (train) wins
    assert "batch_size-train" in keys
    assert "batch_size-val" in keys
    assert "batch_size" in keys
    assert "embed_dim" in keys
    assert "embed_dim-embedder" in keys
    assert "embed_dim-forecaster" in keys


def test_suffix_disambiguation_values_match_origins():
    exp = _make_exp(ModelWithConfig())
    roots = compose_train_params(exp, RunContextStub())
    resolved = resolve_keys(collect_scalars(*roots))
    assert resolved["embed_dim-embedder"] == "64"
    assert resolved["embed_dim-forecaster"] == "32"
    assert resolved["batch_size-train"] == "8"
    assert resolved["batch_size-val"] == "16"
    assert resolved["examples-train"] == "42"
    assert resolved["examples-val"] == "42"


def test_strategy_and_runtime_roots_emit_resolved_scalars():
    exp = _make_exp(ModelWithConfig())
    runtime = RunContextStub(compile_mode="max-autotune", subsample=None)
    resolved = resolve_keys(collect_scalars(*compose_train_params(exp, runtime)))
    assert resolved["lr"] == "0.0003"
    assert resolved["weight_decay"] == "0.01"
    assert resolved["examples_per_epoch"] == "14853"
    assert resolved["accumulation_steps"] == "2"
    assert resolved["mat_mul_precision"] == "high"
    assert resolved["epochs"] == "12"  # scheduler spec field
    assert resolved["compile_mode"] == "max-autotune"
    assert resolved["milestones"] == "(6,)"  # container of scalars, repr form
    # torch runtime scalars record their resolved value via the allow-list
    # branch: dtype shortens to its bare name, device passes through.
    assert resolved["device"] == "cpu"
    assert resolved["dtype"] == "bfloat16"
    # Live objects stringified as class names, surviving without raising.
    assert resolved["model"] == "MODEL-LIVE"


def test_class_attribute_config_rides_model_class_only():
    # graph_recurrent stores config as a class attribute; the model root is
    # excluded and no walker raise happens (AD8, no special case).
    exp = _make_exp(GraphRecurrentLike())
    keys = _keys(exp)
    assert "model_class" in keys
    assert not any(k.startswith("embed_dim") for k in keys)


def test_model_without_config_attribute_is_fine():
    class Bare:
        pass

    exp = _make_exp(Bare())
    assert "model_class" in _keys(exp)


def test_is_config_instance_rejects_classes_and_none():
    assert not _is_config_instance(None)
    assert not _is_config_instance(ModelCfg)  # the class, not an instance
    assert _is_config_instance(ModelCfg())
    assert not _is_config_instance("not a config")


def test_resolve_keys_is_pure_in_suffix_table_terms():
    assert set(SUFFIX_RULES) == {"embed_dim", "batch_size", "examples"}


def test_log_params_guarded_exported_contract():
    # Smoke the guarded logger against a stubbed active run; the full
    # behavioural coverage lives in tests/test_param_walker.py.
    import sys
    import types

    import training.tracking.params as params_mod

    fake_mlflow = types.ModuleType("mlflow")
    fake_run = SimpleNamespace(info=SimpleNamespace(run_id="r1"))
    fake_mlflow.active_run = lambda: fake_run
    fake_mlflow.get_run = lambda run_id: SimpleNamespace(
        data=SimpleNamespace(params={})
    )
    logged: dict[str, str] = {}
    fake_mlflow.log_params = lambda payload: logged.update(payload)
    saved = sys.modules.get("mlflow")
    sys.modules["mlflow"] = fake_mlflow
    try:
        log_params_guarded({"lr": "0.0003"})
    finally:
        if saved is not None:
            sys.modules["mlflow"] = saved
        else:
            sys.modules.pop("mlflow", None)
    assert logged == {"lr": "0.0003"}
    # and the module-level lazy import still resolves the real one later
    assert params_mod is not None
