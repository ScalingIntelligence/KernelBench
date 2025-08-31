import argparse
import os

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO_TOP_DIR, "runs")
KERNEL_EVAL_BUILD_DIR = "/data/user_data/gyeongwk/KernelBench/cache"


def add_inference_args(parser, rl_training=False):
    parser.add_argument("--server_type", type=str, default="vllm")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--vllm_host", type=str, default="localhost") # server_type is vllm
    parser.add_argument("--vllm_port", type=int, default=8081) # server_type is vllm
    if not rl_training:    
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--api_query_interval", type=float, default=0.0)


def add_eval_args(parser):
    parser.add_argument("--hardware", type=str, default="A6000_babel") # GPU hardware type: this should match baseline hardware name
    parser.add_argument("--gpu_arch", type=str, default="Ampere") # GPU architecture: make sure matches hardware type
    parser.add_argument("--num_eval_devices", type=int, default=1) # number of GPUs used for evaluation

    # Kernel compilation
    parser.add_argument("--build_cache_with_cpu", type=bool, default=True)
    parser.add_argument("--num_cpu_workers", type=int, default=1)

    # Kernel evaluation
    parser.add_argument("--num_correct_trials", type=int, default=5)
    parser.add_argument("--num_perf_trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--measure_performance", type=bool, default=True)


def add_logging_args(parser):
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_prompt", type=bool, default=True)
    parser.add_argument("--log_response", type=bool, default=True)
    parser.add_argument("--store_type", type=str, default="local")


def post_process_dataset_args(args):
    range_str = args.subset.strip("()").split(",")
    if range_str[0] != "None":
        args.run_name = args.run_name + "_" + range_str[0] + "_" + range_str[1]
    args.subset = (None, None) if range_str[0] == "None" else (int(range_str[0]), int(range_str[1]))
    return args



def parse_test_time_scaling_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--_tags", type=str, default="test_test_time_scaling")

    parser.add_argument("--run_name", type=str, required=True)

    # Methods
    parser.add_argument("--method", type=str, default="base")
    parser.add_argument("--prompt", type=str, default="regular")
    parser.add_argument("--num_parallel", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--num_best", type=int, default=1)

    # Dataset
    parser.add_argument("--dataset_src", type=str, default="local")
    parser.add_argument("--dataset_name", type=str, default="ScalingIntelligence/KernelBench")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--subset", type=str, default="(None, None)")

    # Inference Server
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    add_inference_args(parser)
    
    # Eval
    parser.add_argument("--eval_mode", type=str, default="local") # should be local
    parser.add_argument("--eval_server_host", type=str, default="localhost") # eval_mode is remote
    parser.add_argument("--eval_server_port", type=int, default=12345) # eval_mode is remote
    add_eval_args(parser)

    add_logging_args(parser)

    args = parser.parse_args()

    # Post processing
    args = post_process_dataset_args(args)
    return args


def parse_evaluation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--_tags", type=str, default="test_evaluation")
    parser.add_argument("--run_name", type=str, required=True) # should match existing run directory with kernels

    # Dataset
    parser.add_argument("--dataset_src", type=str, default="local")
    parser.add_argument("--dataset_name", type=str, default="ScalingIntelligence/KernelBench")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--subset", type=str, default="(None, None)")

    add_eval_args(parser)

    add_logging_args(parser)

    args = parser.parse_args()

    # Post processing
    args = post_process_dataset_args(args)
    args.method = "base"
    args.prompt = "regular"
    return args


def parse_rl_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--_tags", type=str, default="test_rl_training")

    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--multi_turn", action="store_true")

    # Dataset
    parser.add_argument("--dataset_src", type=str, default="local")
    parser.add_argument("--dataset_name", type=str, default="ScalingIntelligence/KernelBench")
 
    # Inference Server
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    add_inference_args(parser)

    # Eval
    parser.add_argument("--eval_mode", type=str, default="remote") # should be remote
    parser.add_argument("--eval_server_host", type=str, default="localhost") # eval_mode is remote
    parser.add_argument("--eval_server_port", type=int, default=12345) # eval_mode is remote
    parser.add_argument("--gpu_offset", type=int, default=2) # number of GPUs used for training (for local eval)
    add_eval_args(parser)

    add_logging_args(parser)


    args = parser.parse_args()

    # Post processing
    args.method = "base" # base prompt for single-turn RL
    args.prompt = "regular" # set as regular for the prompt
    if args.eval_mode == "local":
        args.max_concurrent_eval = args.num_gpu_devices - args.gpu_offset
    else:
        args.max_concurrent_eval = 16 # TODO: make this configurable
    return args


def parse_eval_server_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_src", type=str, default="local")
    parser.add_argument("--dataset_name", type=str, default="ScalingIntelligence/KernelBench")

    parser.add_argument("--port", type=int, default=12345)

    add_eval_args(parser)

    add_logging_args(parser)

    args = parser.parse_args()
    args.method = "base"
    return args


def parse_autorule_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--_tags", type=str, default="test_autorule")

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--level", type=int, required=True)

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    add_inference_args(parser)

    add_logging_args(parser)

    args = parser.parse_args()
    return args