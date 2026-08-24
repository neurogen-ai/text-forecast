from typing import Protocol, override

import torch
from torch import Tensor
from torch.nn import BCELoss
from torch.nn.functional import binary_cross_entropy

from data.datasets.types import TokenBatch

from utils import component

from .loss_protocol import LossFn


class BCEOutput(Protocol):
    probs: Tensor


@component
class BinaryCrossEntropyLoss(BCELoss, LossFn[TokenBatch, BCEOutput]):
    def __init__(self, config):
        super().__init__()
        self.torch_bce_loss = binary_cross_entropy

    @override
    def __call__(self, output: BCEOutput, batch: TokenBatch) -> Tensor:
        # weight is None once datasets emit Optional weights (1.4.6); until
        # then the NaN sentinel marks "unused".
        weight = batch.weight
        return self.torch_bce_loss(
            input=output.probs.squeeze(-1),
            target=batch.y.squeeze(-1),
            weight=weight.squeeze(-1)
            if weight is not None and not torch.any(torch.isnan(weight))
            else None,
        )
