"""Unit tests for the param key scheme (plans/implementation/v2.3.1.md,
v2.3.1-param-walker commit 2). Every leaf emits its bare key; leaves named
in SUFFIX_RULES also emit "<leaf>-<suffix>" when the suffix appears as a
segment of the leaf's origin; the key set is a pure function of the leaves
and the table.
"""

from training.tracking.param_keys import SUFFIX_RULES, resolve_keys
from training.tracking.params import Leaf


def test_bare_keys_from_nested_origins():
    leaves = {
        "strategy.epochs": Leaf(value=3, origin="strategy.epochs"),
        "strategy.optimizer_spec.lr": Leaf(
            value=0.001, origin="strategy.optimizer_spec.lr"
        ),
    }
    assert resolve_keys(leaves) == {"epochs": "3", "lr": "0.001"}


def test_bare_plus_suffix_duplication():
    leaves = {
        "train.batch_size": Leaf(value=32, origin="train.batch_size"),
        "train.examples": Leaf(value=100, origin="train.examples"),
        "val.batch_size": Leaf(value=16, origin="val.batch_size"),
        "val.examples": Leaf(value=50, origin="val.examples"),
    }
    assert resolve_keys(leaves) == {
        "batch_size": "32",
        "batch_size-train": "32",
        "batch_size-val": "16",
        "examples": "100",
        "examples-train": "100",
        "examples-val": "50",
    }


def test_tabled_differing_values_first_declaration_wins_bare_key():
    leaves = {
        "model.config.embedder.embed_dim": Leaf(
            value=8, origin="model.config.embedder.embed_dim"
        ),
        "model.config.forecaster.embed_dim": Leaf(
            value=16, origin="model.config.forecaster.embed_dim"
        ),
    }
    assert resolve_keys(leaves) == {
        "embed_dim": "8",
        "embed_dim-embedder": "8",
        "embed_dim-forecaster": "16",
    }


def test_suffix_not_in_origin_emits_bare_key_only():
    leaves = {"model.embed_dim": Leaf(value=4, origin="model.embed_dim")}
    assert resolve_keys(leaves) == {"embed_dim": "4"}


def test_same_value_bare_key_duplicates_collapse():
    leaves = {
        "a.x": Leaf(value="1", origin="a.x"),
        "b.x": Leaf(value="1", origin="b.x"),
    }
    assert resolve_keys(leaves) == {"x": "1"}


def test_key_set_is_identical_across_runs():
    def build() -> dict[str, str]:
        return resolve_keys(
            {
                "train.batch_size": Leaf(value=32, origin="train.batch_size"),
                "val.batch_size": Leaf(value=16, origin="val.batch_size"),
                "model.config.embedder.embed_dim": Leaf(
                    value=8, origin="model.config.embedder.embed_dim"
                ),
            }
        )

    assert build() == build()


def test_suffix_rules_table_shape():
    assert set(SUFFIX_RULES) == {"embed_dim", "batch_size", "examples"}
