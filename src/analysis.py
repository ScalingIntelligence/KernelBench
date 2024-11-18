################################################################################
# Helpers for Analysis
################################################################################
import numpy as np


def pass_at_k(n, c, k):
    """
    A numerically stable script for calculating an unbiased estimate of pass@k
    Referenced from HumanEval: https://arxiv.org/abs/2107.03374
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
