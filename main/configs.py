import argparse
import os

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(rl_training=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--_tags", type=str, default="test")

    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--runs_dir", type=str, default=os.path.join(REPO_TOP_DIR, "runs"))

    # Methods
    if not rl_training:
        parser.add_argument("--method", type=str, default="base")
        parser.add_argument("--prompt", type=str, default="regular")
        parser.add_argument("--num_parallel", type=int, default=1)
        parser.add_argument("--num_samples", type=int, default=1)
        parser.add_argument("--num_iterations", type=int, default=1)
        parser.add_argument("--num_best", type=int, default=1)

    # Dataset
    parser.add_argument("--dataset_src", type=str, default="local")
    # parser.add_argument("--dataset_name", type=str, default="ScalingIntelligence/KernelBench")
    if not rl_training:
        parser.add_argument("--level", type=int, required=True)
        parser.add_argument("--subset", type=str, default="(None, None)")

    # Inference Server
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    if not rl_training:
        parser.add_argument("--server_type", type=str, default="openai")
        parser.add_argument("--max_tokens", type=int, default=4096)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--api_query_interval", type=float, default=0.0)

    # Eval
    parser.add_argument("--eval_mode", type=str, default="local")
    parser.add_argument("--gpu_arch", type=str, default="Ampere")
    parser.add_argument("--num_gpu_devices", type=int, default=1)
    if rl_training:
        parser.add_argument("--eval_device", type=str, default="cuda:1")

    parser.add_argument("--build_cache_with_cpu", type=bool, default=True)
    parser.add_argument("--num_cpu_workers", type=int, default=1)
    parser.add_argument("--kernel_eval_build_dir", type=str, default=os.path.join(REPO_TOP_DIR, "cache"))

    parser.add_argument("--num_correct_trials", type=int, default=5)
    parser.add_argument("--num_perf_trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--measure_performance", type=bool, default=True)

    # Logging
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--log_prompt", type=bool, default=True)
    parser.add_argument("--log_response", type=bool, default=True)
    parser.add_argument("--store_type", type=str, default="local")

    args = parser.parse_args()

    # Post processing
    args.gpu_arch = args.gpu_arch.split(",")
    
    if not rl_training:
        range_str = args.subset.strip("()").split(",")
        if range_str[0] != "None":
            args.run_name = args.run_name + "_" + range_str[0] + "_" + range_str[1]
        args.subset = (None, None) if range_str[0] == "None" else (int(range_str[0]), int(range_str[1]))
    return args