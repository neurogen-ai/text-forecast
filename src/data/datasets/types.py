"""Canonical batch and output types shared by datasets, models, strategies,
losses, and trackers (plan 1.4.1)."""

from typing import NamedTuple

from torch import Tensor


class TokenBatch(NamedTuple):
    """Canonical batch for token-id sequence classification.

    ``__getitem__`` returns per-item tensors of shape ``(T,)``; the DataLoader
    collates them into the batch shapes below.

    Attributes:
        id: ``(B,)`` row identifiers.
        x: ``(B, T)`` long token ids, padded to ``max_len`` with ``pad_token_id``.
        y: ``(B, 1)`` float targets in ``{0, 1}``.
        mask: ``(B, T)`` bool key-padding mask, True where a token is real
            (not padding). Attention consumers broadcast it to head shape
            themselves, e.g. ``mask[:, None, None, :]`` for SDPA.
        weight: ``(B,)`` float loss weights, or None when the dataset has no
            weights configured.
    """

    id: Tensor
    x: Tensor
    y: Tensor
    mask: Tensor
    weight: Tensor | None


class CitationGraphDatasetOutput(NamedTuple):
    id: Tensor
    x: Tensor
    graph_x: Tensor
    y: Tensor
    weight: Tensor
    mask: Tensor
    graph_x_mask: Tensor
