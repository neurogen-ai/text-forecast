from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
from torch import Tensor

from data.datasets.types import CitationGraphDatasetOutput

from .base import Formater


def _default_format_x(x: Tensor) -> Tensor:
    return x.long()


def _default_format_y(y: Tensor) -> Tensor:
    return y


@dataclass(frozen=True, kw_only=True)
class CitationGraphFormater(Formater[Mapping[str, Any], CitationGraphDatasetOutput]):
    """Default per-row formatter for ``CitationGraphDataset``.

    Mirrors the historical ``CitationGraphDataset.__getitem__`` behaviour:
    flattening, padding to ``max_len`` / ``graph_max_len`` with the token id,
    and mask construction.  The historical ``graph_x = format_x(x)`` path is
    preserved as the default ``format_graph_x`` so current behaviour is
    unchanged.
    """

    max_len: int
    graph_max_len: int
    pad_token_id: int
    pad: bool = True
    truncate: bool = True
    truncate_method: Literal["truncate", "drop"] = "drop"
    return_mask: bool = True
    return_id: bool = False
    weights: Tensor | None = None
    format_x: Callable[[Tensor], Tensor] = _default_format_x
    format_y: Callable[[Tensor], Tensor] = _default_format_y
    format_graph_x: Callable[[Tensor, Tensor], Tensor] | None = None

    def _resolve_format_graph_x(self) -> Callable[[Tensor, Tensor], Tensor]:
        # Default preserves historical ``graph_x = self._format_x(x)``.
        if self.format_graph_x is not None:
            return self.format_graph_x
        return lambda graph_x, x: self.format_x(x)

    def __call__(self, row: Mapping[str, Any]) -> CitationGraphDatasetOutput:
        id_out: Any = torch.tensor(float("nan"))
        mask = torch.tensor(float("nan"))
        graph_x_mask = torch.tensor(float("nan"))
        weight = torch.tensor(float("nan"))

        x: Tensor = torch.tensor(row["x"], dtype=torch.float32).flatten()
        x = self.format_x(x)

        if self.pad and x.size(0) < self.max_len:
            x = nn.functional.pad(
                x,
                (0, self.max_len - x.size(0)),
                value=self.pad_token_id,
            )

        if self.truncate and x.size(0) > self.max_len:
            x = x[: self.max_len]

        graph_x: Tensor = torch.tensor(row["graph_x"], dtype=torch.float32).flatten()
        # Historical behaviour formats graph_x using x.
        graph_x = self._resolve_format_graph_x()(graph_x, x)

        if self.pad and graph_x.size(0) < self.graph_max_len:
            graph_x = nn.functional.pad(
                graph_x,
                (0, self.graph_max_len - graph_x.size(0)),
                value=self.pad_token_id,
            )

        if self.truncate and graph_x.size(0) > self.graph_max_len:
            graph_x = graph_x[: self.graph_max_len]

        y: Tensor = torch.tensor(row["y"], dtype=torch.float32).flatten()
        y = self.format_y(y)

        if self.return_id:
            id_out = row["id"]

        if self.weights is not None:
            weight = torch.tensor(
                [
                    self.weights[target.long()].item()
                    for target in torch.atleast_1d(y)
                ],
                dtype=torch.float32,
            ).flatten()

        if self.return_mask:
            mask = (x != self.pad_token_id).bool()
            graph_x_mask = (graph_x != self.pad_token_id).bool()

        return CitationGraphDatasetOutput(
            id=id_out,
            x=x,
            graph_x=graph_x,
            y=y,
            mask=mask,
            graph_x_mask=graph_x_mask,
            weight=weight,
        )
