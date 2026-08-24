"""Canonical model protocol (plan 1.4.2).

Every ``@component`` model must satisfy this protocol and declare it
explicitly on the class so basedpyright flags signature drift:

- a ``config`` **class attribute** holding the model's config type (used by
  experiment builders to introspect defaults), plus an instance attribute of
  the same name holding the materialised config;
- ``forward(self, batch)`` taking one canonical batch NamedTuple from
  ``data.datasets.types`` and returning an output exposing ``.logits``.

Legacy models that are not worth migrating carry a ``# legacy:`` header
comment instead and stay off the conformance test; exclusion is visible,
never accidental.
"""

from typing import Protocol


class Model[T_Config, T_Batch, T_Out](Protocol):
    """A trainable model: ``config`` + ``forward(batch) -> output``."""

    config: type[T_Config]

    def forward(self, batch: T_Batch) -> T_Out: ...
