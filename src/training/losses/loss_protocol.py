from typing import Protocol


class LossFn[T_Batch, T_Output](Protocol):
    """A loss consumes model ``output`` and its ``batch``.

    Binary-classification losses take **logits only** (sub-plan 1.4.3):
    models must not apply sigmoid inside ``forward``. Multi-class losses
    consume logits too; any softmax happens inside the loss.
    """

    def forward(self, batch: T_Batch, output: T_Output) -> float: ...
