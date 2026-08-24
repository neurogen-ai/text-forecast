from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from torch import Tensor

from data.datasets.types import CitationGraphDatasetOutput

from .base import Formater


def _default_format_x(x: Tensor) -> Tensor:
    return x


def _graph_format_y(y: Tensor) -> Tensor:
    return torch.tensor(y > 15, dtype=torch.float32)


@dataclass(frozen=True, kw_only=True)
class GraphFormater(Formater[Mapping[str, Any], CitationGraphDatasetOutput]):
    """Default per-row formatter for ``GraphDataset``.

    Mirrors the historical ``GraphDataset.__getitem__`` behaviour exactly:
    tensor conversion, optional x truncation, graph_x padding to ``top_k`` with
    zeros, graph_x truncation, and per-class weight lookup.  Column selection
    remains the dataset's responsibility.
    """

    max_len: int
    graph_max_len: int
    top_k: int
    pad: bool = True
    truncate: bool = True
    truncate_method: Literal["truncate", "drop"] = "drop"
    return_mask: bool = True
    return_id: bool = False
    weights: Tensor | None = None
    format_x: Callable[[Tensor], Tensor] = _default_format_x
    format_y: Callable[[Tensor], Tensor] = _graph_format_y

    def __call__(self, row: Mapping[str, Any]) -> CitationGraphDatasetOutput:
        # Use NaN placeholders to match the original layout.
        id_out: Any = torch.tensor(float("nan"))
        mask = torch.tensor(float("nan"))
        graph_x_mask = torch.tensor(float("nan"))
        weight = torch.tensor(float("nan"))

        x: Tensor = torch.tensor(row["x"], dtype=torch.float32)
        x = self.format_x(x)

        if self.truncate and x.size(0) > self.max_len:
            x = x[: self.max_len]

        graph_x: Tensor = torch.tensor(row["graph_x"], dtype=torch.float32)
        # Historical GraphDataset behaviour: graph_x is not formatted.

        if self.pad and graph_x.size(0) < self.top_k:
            pad_len = self.top_k - graph_x.size(0)
            padding = torch.zeros(
                (pad_len, graph_x.size(1)),
                dtype=graph_x.dtype,
                device=graph_x.device,
            )
            graph_x = torch.cat([graph_x, padding], dim=0)

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
            # Historical GraphDataset behaviour.
            graph_x_mask = torch.zeros((self.top_k, 1))
            graph_x_mask[: len(row["x"])] = 1

        return CitationGraphDatasetOutput(
            id=id_out,
            x=x,
            graph_x=graph_x,
            y=y,
            mask=mask,
            graph_x_mask=graph_x_mask,
            weight=weight,
        )
