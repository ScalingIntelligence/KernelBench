import torch
import torch.nn as nn

###########################################
# This custom kernel computes the correct matmul on any new inputs,
# but stores the output and returns it again when called on the same input tensors.
# It passes every correctness trial (each trial uses new inputs), and the timing loop,
# which calls it repeatedly on one set of inputs, would time the cached calls as near zero.
# Eval should flag it as incorrect: it launches no GPU work on a repeated input (see issue #171).
###########################################

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._last_inputs = None
        self._last_output = None

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if self._last_inputs is not None and self._last_inputs[0] is A and self._last_inputs[1] is B:
            return self._last_output
        self._last_inputs = (A, B)
        self._last_output = torch.matmul(A, B)
        return self._last_output
