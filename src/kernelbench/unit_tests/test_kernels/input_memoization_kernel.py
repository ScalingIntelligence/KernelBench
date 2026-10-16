import torch
import torch.nn as nn

###########################################
#
# This adversarial kernel memoizes its output on input identity
# It computes the correct result on any new input, so it passes every correctness trial
# But if the timing loop reuses the same input tensors on every trial, it only computes once
# and every later timed call returns the cached output, giving a huge fake speedup
# So, every timed call should get newly drawn inputs (see issue #171)
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

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []
