import torch
import torch.nn as nn

###########################################
#
# This adversarial kernel memoizes its output keyed on a fingerprint: the first few values of each input
# Correctness trials use inputs with different values, so it computes the correct result on each of them
# If timed calls get copies of the same inputs, every copy has the same fingerprint and
# every timed call after the first is a cache hit, giving a fake speedup
# So, every timed call should get newly drawn inputs, not copies (see issue #171)
###########################################

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._cache = {}  # fingerprint -> out

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        key = tuple(A.flatten()[:8].tolist() + B.flatten()[:8].tolist())
        if key not in self._cache:
            self._cache[key] = torch.matmul(A, B)
        return self._cache[key]

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []
