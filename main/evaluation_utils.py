import os
import socket
import pickle
import torch
import json
import time
from tqdm import tqdm
import multiprocessing as mp
import yaml
from dataclasses import dataclass

from src.compile import batch_compile, remove_cache_dir
from src.eval import eval_kernel_against_ref, eval_reference_kernel, KernelExecResult, check_metadata_serializable_all_types
from src.utils import set_gpu_arch

from configs import parse_evaluation_args, RUNS_DIR, KERNEL_EVAL_BUILD_DIR
from dataset import construct_kernelbench_dataset, fetch_ref_arch_from_level_problem_id
from utils import WorkArgs
from run_utils import fetch_kernel_from_disk, check_if_eval_exists_local
from evaluation_utils_modal import evalaute_single_sample_modal


@dataclass
class EvaluationWorkArgs:
    level: int
    problem_id: int
    sample_id: int
    device: torch.device

def serialize_work_args(work_args: EvaluationWorkArgs):
    """Serialize EvaluationWorkArgs for network transmission"""
    return {
        'level': work_args.level,
        'problem_id': work_args.problem_id,
        'sample_id': work_args.sample_id,
        'device': str(work_args.device)  # Convert device to string for serialization
    }


def deserialize_work_args(data: dict) -> EvaluationWorkArgs:
    """Deserialize data back to EvaluationWorkArgs"""
    return EvaluationWorkArgs(
        level=data['level'],
        problem_id=data['problem_id'],
        sample_id=data['sample_id'],
        device=torch.device(data['device'])
    )


def evaluate_single_sample_worker(work_args: EvaluationWorkArgs, configs, run_dir: str, kernel_src=None, kernel_name=None) -> KernelExecResult | None:
    """
    Evaluate a single sample on a single GPU
    """
    level, problem_id, sample_id, device = (
        work_args.level,
        work_args.problem_id,
        work_args.sample_id,
        work_args.device,
    )
    # Fetch reference architecture from problem directory
    ref_arch_src, ref_arch_name = fetch_ref_arch_from_level_problem_id(level, problem_id, configs.dataset_src)

    # Fetch kernel from disk
    if kernel_src is None:
        kernel_src, kernel_name = fetch_kernel_from_disk(run_dir, level, problem_id, sample_id)
    assert kernel_src is not None, f"Kernel not found for problem {problem_id} sample {sample_id}"

    build_dir = os.path.join(KERNEL_EVAL_BUILD_DIR, configs.run_name, f"level_{level}", f"{problem_id}", f"{sample_id}")

    try: 
        if configs.method == "METR" and sample_id == 0:
            eval_result = eval_reference_kernel(
                original_model_src=ref_arch_src,
                original_model_name=ref_arch_name,
                verbose=configs.verbose,
                device=device,
            )
        else:
            eval_result = eval_kernel_against_ref(
                original_model_src=ref_arch_src,
                original_model_name=ref_arch_name,
                custom_model_src=kernel_src,
                custom_model_name=kernel_name,
                measure_performance=configs.measure_performance,
                verbose=configs.verbose,    
                num_correct_trials=configs.num_correct_trials,
                num_perf_trials=configs.num_perf_trials,
                build_dir=build_dir,
                device=device,
            )
        return eval_result
    except Exception as e:
        print(
            f"[WARNING] Last level catch on {sample_id}: Some issue evaluating for kernel: {e} "
        )
        if "CUDA error" in str(e):
            # NOTE: count this as compilation failure as it is not runnable code
            metadata = {
                "cuda_error": f"CUDA Error: {str(e)}",
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }  # log this for debugging as this usually signifies illegal memory access
            eval_result = KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )
            return eval_result
        else:
            metadata = {"other_error": f"error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        } # for debugging
            eval_result = KernelExecResult(compiled=False, correctness=False, 
                                                metadata=metadata)
            return eval_result


def evaluate_single_sample_in_separate_process(work_args: EvaluationWorkArgs, configs, run_dir: str, kernel_src=None, kernel_name=None) -> KernelExecResult | None:
    """
    Evaluate a single sample in a separate process
    """
    level, problem_id, sample_id, device = (
        work_args.level,
        work_args.problem_id,
        work_args.sample_id,
        work_args.device,
    )
    
    
    # Create argument tuple for the process
    args_tuple = (work_args, configs, run_dir, kernel_src, kernel_name)
    
    # Run evaluation in separate process with timeout
    with mp.Pool(1) as pool:
        try:
            result = pool.apply_async(evaluate_single_sample_worker, args_tuple)
            eval_result = result.get(timeout=300)  # 5 minute timeout
            return eval_result
        except mp.TimeoutError:
            metadata = {
                "other_error": "Evaluation timed out after 5 minutes",
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }
            return KernelExecResult(
                compiled=False, 
                correctness=False, 
                metadata=metadata
            )
        except Exception as e:
            metadata = {
                "other_error": f"Pool error: {str(e)}",
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }
            return KernelExecResult(
                compiled=False, 
                correctness=False, 
                metadata=metadata
            )


def add_to_eval_results_file(level: int, problem_id: int, sample_id: int, eval_result: KernelExecResult, eval_file_path: str):
    """
    Add evaluation result to eval results file
    TODO: migrate database support
    """
    # Load existing results if file exists
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = {}
    
    # Add new result
    if str(level) not in eval_results:
        eval_results[str(level)] = {}

    if str(problem_id) not in eval_results[str(level)]:
        eval_results[str(level)][str(problem_id)] = {}
    
    eval_results[str(level)][str(problem_id)][str(sample_id)] = {
        'problem_id': problem_id,
        'sample_id': sample_id,
        'compiled': eval_result.compiled,
        'correctness': eval_result.correctness,
        'metadata': check_metadata_serializable_all_types(eval_result.metadata),
        'runtime': eval_result.runtime,
        'runtime_stats': eval_result.runtime_stats,
    }
    
    # Write updated results back to file
    if not os.path.exists(eval_file_path):
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        
    with open(eval_file_path, "w") as f:
        json.dump(eval_results, f, indent=2)


def write_eval_result_for_sample(level: int, problem_id: int, sample_id: int, eval_result: KernelExecResult, run_dir):
    file_path = os.path.join(run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_eval_result.json")
    with open(file_path, "w") as f:
        json.dump(eval_result, f, indent=2)


def send_evaluation_request(host: str, port: int, work_args: EvaluationWorkArgs, kernel_src: str = None, kernel_name: str = None):
    """
    Send an evaluation request to the server and receive the result.
    
    Args:
        host: Server hostname (usually 'localhost')
        port: Server port number
        work_args: EvaluationWorkArgs object
        kernel_src: Optional kernel source code
        kernel_name: Optional kernel name
    
    Returns:
        KernelExecResult object from the server
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Connect to the server
        client_socket.connect((host, port))
        print(f"Connected to evaluation server at {host}:{port}")
        
        # Prepare the request
        request = {
            'work_args': serialize_work_args(work_args),
            'kernel_src': kernel_src,
            'kernel_name': kernel_name
        }
        
        # Send the request
        request_data = pickle.dumps(request)
        client_socket.sendall(request_data)
        
        # Receive the response
        response_data = b""
        while True:
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            response_data += chunk
        
        # Deserialize the response
        result = pickle.loads(response_data)
        return result
        
    except Exception as e:
        print(f"Error communicating with server: {e}")
        return None
    finally:
        client_socket.close()


def check_server_status(host: str, port: int):
    """
    Check if the server is running and get basic status information.
    
    Args:
        host: Server hostname
        port: Server port number
    
    Returns:
        True if server is reachable, False otherwise
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(5)  # 5 second timeout
    
    try:
        client_socket.connect((host, port))
        print(f"Server at {host}:{port} is reachable")
        return True
    except Exception as e:
        print(f"Cannot connect to server at {host}:{port}: {e}")
        return False
    finally:
        client_socket.close()


def check_eval_status(config):
    if config.eval_method == "local":
        return True
    elif config.eval_method == "remote":
        return check_server_status(config.eval_server_host, config.eval_server_port)
    elif config.eval_method == "modal":
        return True
    else:
        raise ValueError(f"Invalid evaluation method: {config.eval_method}")


def evaluate_single_sample(work_args: EvaluationWorkArgs, configs, run_dir: str, kernel_src=None, kernel_name=None) -> KernelExecResult | None:
    """
    Evaluate a single sample using the specified evaluation method
    """
    if config.eval_method == "local":
        return evaluate_single_sample_worker(work_args, configs, run_dir, kernel_src, kernel_name)
    elif config.eval_method == "remote":
        return send_evaluation_request(config.eval_server_host, config.eval_server_port, work_args, kernel_src, kernel_name)
    elif config.eval_method == "modal":
        return evalaute_single_sample_modal(work_args, configs, run_dir, kernel_src, kernel_name)


def batch_eval(
    total_work: list[WorkArgs],
    config: dict,
    run_dir: str,
    eval_file_path: str,
):
    """
    Batch evaluation across multiple GPUs, do batch_size of work one on each GPU all at once
    We put in time out for each batch, consider trying again with larger time out if it didn't finish building.
    Cache directory is removed if evaluation times out or fails
    NOTE: Only for local evaluation
    """
    total_work = [work for work in total_work if not check_if_eval_exists_local(work.level, work.problem_id, work.sample_id, eval_file_path)]

    # Build Cache on CPU as that is faster
    if config.build_cache_with_cpu:
        compilation_results = batch_compile([(arg.level, arg.problem_id, arg.sample_id) for arg in total_work], vars(config))

    # construct a list of work args
    batch_size = config.num_gpu_devices

    with tqdm(total=len(total_work), desc="Evaluation Progress") as pbar:

        while len(total_work) > 0:
            curr_work_batch = total_work[:batch_size]
            total_work = total_work[batch_size:]  # pop the first batch_size elements
            print(
                f"[Curr Batch] {len(curr_work_batch)} tasks over {config.num_gpu_devices} GPUs; [Total Work left] {len(total_work)}"
            )
            assert len(curr_work_batch) <= batch_size, f"Current batch size {len(curr_work_batch)} is greater than the number of GPUs {batch_size}"

            with mp.Pool(batch_size) as pool:

                work_args = [
                    (
                        EvaluationWorkArgs(
                            level=work_arg.level,
                            problem_id=work_arg.problem_id,
                            sample_id=work_arg.sample_id,
                            device=torch.device(f"cuda:{i%batch_size}"),
                        ),
                        config,
                        run_dir,
                    )
                    for i, work_arg in enumerate(curr_work_batch)
                ]

                start_time = time.time()

                async_results = []
                for work_arg in work_args:
                    async_results.append(
                        pool.apply_async(evaluate_single_sample_worker, work_arg)
                    )
            
                # Collect results with a batch timeout
                results = []
                batch_timeout = config.timeout
                for i, async_result in enumerate(async_results):
                    work_arg = curr_work_batch[i]
                    level, problem_id, sample_id = work_arg.level, work_arg.problem_id, work_arg.sample_id

                    try:
                        elapsed_time = time.time() - start_time
                        remaining_time = max(0, batch_timeout - elapsed_time)
                        result = async_result.get(timeout=remaining_time)
                        results.append((level, problem_id, sample_id, result))
                        
                    except mp.TimeoutError:
                        print(
                            f"[WARNING] Evaluation TIMED OUT for Problem ID: {problem_id}, Sample ID: {sample_id}"
                        )
                        result = KernelExecResult(compiled=False, correctness=False, metadata={"other_error": "timeout"})
                        results.append((level, problem_id, sample_id, result))
                    
                        remove_cache_dir(vars(config), level, problem_id, sample_id)
                    except Exception as e:
                        print(
                            f"[ERROR] Evaluation FAILED for Problem ID: {problem_id}, Sample ID: {sample_id}: {str(e)}"
                        )
                        result = KernelExecResult(compiled=False, correctness=False, metadata={"other_error": str(e)})
                        results.append((level, problem_id, sample_id, result))
                        remove_cache_dir(vars(config), level, problem_id, sample_id)

                end_time = time.time()

                # current batch summary
                for level, problem_id, sample_id, result in results:
                    print("-" * 128)
                    print(
                        f"[Eval Result] Problem ID: {problem_id}, Sample ID: {sample_id}"
                    )
                    print(result)

                    # add all the batch results here to avoid file race condition
                    # add to eval result if valid result
                    if result is not None:
                        add_to_eval_results_file(level, problem_id, sample_id, result, eval_file_path)

                if config.verbose:
                    print("-" * 128)
                    print(
                        f"[Curr batch] Evaluation took {end_time - start_time:.2f} seconds"
                )

                pbar.update(len(curr_work_batch))



if __name__ == "__main__":
    # Just run evaluation for already generated kernels
    config = parse_evaluation_args()
    # wandb.init(project="KernelBench", entity="j1mk1m", tags=[config.run_name, config.method, config.prompt, str(config.level), config.model_name])

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. Evaluation requires GPU.")

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    # 1. Set up
    # Set up dataset
    curr_level_dataset = construct_kernelbench_dataset(config.level)

    # set up run directory
    run_dir = os.path.join(RUNS_DIR, config.run_name)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(config), f)
 
    # set GPU arch to configure what target to build for
    set_gpu_arch(config.gpu_arch)
    assert config.num_gpu_devices <= torch.cuda.device_count(), f"Number of GPUs requested ({config.num_gpu_devices}) is greater than the number of available GPUs ({torch.cuda.device_count()})"

    eval_file_path = os.path.join(run_dir, f"eval_results.json")

    total_work = [WorkArgs(level=config.level, problem_id=problem_id, sample_id=sid) for problem_id in range(1, len(curr_level_dataset) + 1) for sid in range(1)] # TODO: change accordingly

    batch_eval(total_work, config, run_dir, eval_file_path)

    
