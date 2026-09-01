import math

from bench_flash_attn import gen_bench_config, run_bench


SEQLEN_SWEEP = [512, 1024, 2048, 4096, 8192, 16384]
BASE_SEQLEN = 512
BASE_BATCH = 56
NUM_TESTS = 10


def format_seqlen(seqlen: int) -> str:
    return f"{seqlen // 1024}k" if seqlen % 1024 == 0 else str(seqlen)


def batch_for_seqlen(seqlen: int) -> int:
    target_tokens = BASE_BATCH * BASE_SEQLEN
    return max(1, math.ceil(target_tokens / seqlen))


def main() -> None:
    base_case = gen_bench_config(
        seqlen_q_sweep=[BASE_SEQLEN],
        seqlen_kv_sweep=[BASE_SEQLEN],
    )[0]

    target_tokens = BASE_BATCH * BASE_SEQLEN
    print(
        "token_sweep: "
        f"base_batch={BASE_BATCH}, base_seqlen={BASE_SEQLEN}, "
        f"target_tokens={target_tokens}"
    )

    for seqlen in SEQLEN_SWEEP:
        batch_size = batch_for_seqlen(seqlen)
        case = dict(base_case)
        case.update(
            {
                "name": (
                    f"tokensweep_qkv{format_seqlen(seqlen)}"
                    f"_b{batch_size}_tok{batch_size * seqlen}"
                ),
                "batch_size": batch_size,
                "seqlen_q": [seqlen] * batch_size,
                "seqlen_kv": [seqlen] * batch_size,
                "is_causal": True,
            }
        )
        run_bench(num_tests=NUM_TESTS, **case)


if __name__ == "__main__":
    main()
