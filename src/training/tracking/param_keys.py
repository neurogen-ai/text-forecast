"""Param key scheme driven by an explicit rename table.

The rename table owns every non-bare key (AD10). A new ambiguous leaf or a
key rename is one table row; no caller edits, and every suffixed key is
visible in one list. Contract: plans/implementation/v2.3.1.md (Foundational
contracts).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .params import Leaf

# Leaf name -> suffixes to emit when a segment of the leaf's origin matches.
SUFFIX_RULES: dict[str, tuple[str, ...]] = {
    "embed_dim": ("embedder", "forecaster"),
    "batch_size": ("train", "val"),
    "examples": ("train", "val"),
}


def resolve_keys(leaves: dict[str, "Leaf"]) -> dict[str, str]:
    """Map every leaf to the param keys it emits.

    Every leaf emits its bare key (the last segment of its origin). A leaf
    named in SUFFIX_RULES also emits "<leaf>-<suffix>" for each suffix that
    appears as a segment of its origin, so the key set is a pure function
    of the leaves and the table and is identical across runs. For a tabled
    leaf with differing values, the bare key carries the value of the first
    origin in walker declaration order; each origin's value also carries
    its suffixed key. Unknown bare-key collisions surface as the walker's
    KeyError, not here.
    """
    keys: dict[str, str] = {}
    for leaf in leaves.values():
        bare_key = leaf.origin.rsplit(".", 1)[-1]
        value = str(leaf.value)
        if bare_key not in keys:
            keys[bare_key] = value
        for suffix in SUFFIX_RULES.get(bare_key, ()):
            segments = leaf.origin.split(".")
            if suffix in segments:
                keys[f"{bare_key}-{suffix}"] = value
    return keys
