from typing import Protocol, override

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits

from data.datasets.types import TokenBatch

from utils import component

from .loss_protocol import LossFn


class BCEOutput(Protocol):
    """Logits-only contract (sub-plan 1.4.3). The loss applies the sigmoid
    internally via ``binary_cross_entropy_with_logits``, which is numerically
    stabler than sigmoid followed by plain BCE.
    """

    logits: Tensor


@component
class BinaryCrossEntropyLoss(Module, LossFn[TokenBatch, BCEOutput]):
    def __init__(self, config):
        super().__init__()

    @override
    def __call__(self, output: BCEOutput, batch: TokenBatch) -> Tensor:
        # weight is None once datasets emit Optional weights (1.4.6); until
        # then the NaN sentinel marks "unused".
        weight = batch.weight
        return binary_cross_entropy_with_logits(
            input=output.logits.squeeze(-1),
            target=batch.y.squeeze(-1),
            weight=weight.squeeze(-1)
            if weight is not None and not torch.any(torch.isnan(weight))
            else None,
        )
