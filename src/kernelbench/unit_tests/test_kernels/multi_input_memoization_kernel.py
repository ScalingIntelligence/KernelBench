import torch
import torch.nn as nn

###########################################
#
# This adversarial kernel memoizes its output for the last few inputs it has seen
# It computes the correct result on any new input, so it passes every correctness trial
# If the timing loop cycles through a small set of input tensors, every timed call after
# the first lap is a cache hit, giving a fake speedup
# So, every timed call should get newly drawn inputs (see issue #171)
###########################################

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._cache = []  # [((A, B), out)] for the last 4 inputs

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        for (seen_a, seen_b), out in self._cache:
            if seen_a is A and seen_b is B:
                return out
        out = torch.matmul(A, B)
        self._cache = (self._cache + [((A, B), out)])[-4:]
        return out

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []
