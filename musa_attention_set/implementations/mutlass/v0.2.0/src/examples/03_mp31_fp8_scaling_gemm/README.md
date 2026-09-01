# FP8 Groupwise-Scaling GEMM

## Double Accumulator algorithm

In the double accumulator algorithm, we maintain two accumulators. One is used to store the final result, and the other is used for the scaled accumulation of groups.

Due to the limitation of the total number of registers, compared with the standard FP8 GEMM, the tile shape in this method will be halved.


## Iterative algorithm

In the iterative algorithm, we continuously maintain and update the group scaling coefficients and apply them to the same accumulator. Therefore, we can use the same tile shape as the standard FP8 GEMM to further improve performance.

It can be simply described by the following formula.

$S_0 \cdot A_0B_0 + S_1\cdot A_1B_1+S_2\cdot A_2B_2=((\frac{S_0}{S_1}\cdot A_0B_0 + A_1B_1)\cdot \frac{S_1}{S_2} +A_2B_2)\cdot S_2$

We will release the implementation of this algorithm in the future.

