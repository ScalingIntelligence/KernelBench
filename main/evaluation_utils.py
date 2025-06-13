import os
import torch
import json
import time
from tqdm import tqdm
import multiprocessing as mp

from src.compile import batch_compile, remove_cache_dir
from src.eval import eval_kernel_against_ref, KernelExecResult, check_metadata_serializable_all_types

from configs import TestTimeScalingConfig
from utils import WorkArgs, EvaluationWorkArgs, fetch_ref_arch_from_problem_id, fetch_kernel_from_disk, check_if_eval_exists_local


def evaluate_single_sample(work_args: EvaluationWorkArgs, configs: TestTimeScalingConfig, dataset, run_dir: str) -> KernelExecResult | None:
    """
    Evaluate a single sample on a single GPU
    """
    problem_id, sample_id, device = (
        work_args.problem_id,
        work_args.sample_id,
        work_args.device,
    )
    # Fetch reference architecture from problem directory
    ref_arch_src = fetch_ref_arch_from_problem_id(dataset, problem_id, configs.dataset_src)

    # Fetch kernel from disk
    kernel_src = fetch_kernel_from_disk(run_dir, configs.level, problem_id, sample_id)

    assert kernel_src is not None, f"Kernel not found for problem {problem_id} sample {sample_id}"

    build_dir = os.path.join(configs.kernel_eval_build_dir, configs.run_name, f"{problem_id}", f"{sample_id}")

    try: 
        eval_result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
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


def add_to_eval_results_file(problem_id: int, sample_id: int, eval_result: KernelExecResult, eval_file_path: str):
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
    if str(problem_id) not in eval_results:
        eval_results[str(problem_id)] = {}
    
    eval_results[str(problem_id)][str(sample_id)] = {
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
        json.dump(eval_results, f)


def batch_eval(
    total_work: list[WorkArgs],
    config: TestTimeScalingConfig,
    dataset,
    run_dir: str,
    eval_file_path: str,
):
    """
    Batch evaluation across multiple GPUs, do batch_size of work one on each GPU all at once
    We put in time out for each batch, consider trying again with larger time out if it didn't finish building.
    Cache directory is removed if evaluation times out or fails
    """
    total_work = [work for work in total_work if not check_if_eval_exists_local(work.problem_id, work.sample_id, eval_file_path)]

    # Build Cache on CPU as that is faster
    if config.build_cache_with_cpu:
        compilation_results = batch_compile([(arg.problem_id, arg.sample_id) for arg in total_work], config.to_dict())

    # construct a list of work args
    batch_size = config.num_gpu_devices

    with tqdm(total=len(total_work), desc="Processing batches") as pbar:

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
                            problem_id=work_arg.problem_id,
                            sample_id=work_arg.sample_id,
                            device=torch.device(f"cuda:{i%batch_size}"),
                        ),
                        config,
                        dataset,
                        run_dir,
                    )
                    for i, work_arg in enumerate(curr_work_batch)
                ]

                start_time = time.time()

                async_results = []
                for work_arg in work_args:
                    async_results.append(
                        pool.apply_async(evaluate_single_sample, work_arg)
                    )
            
                # Collect results with a batch timeout
                results = []
                batch_timeout = config.timeout
                for i, async_result in enumerate(async_results):
                    work_arg = curr_work_batch[i]
                    problem_id, sample_id = work_arg.problem_id, work_arg.sample_id

                    try:
                        elapsed_time = time.time() - start_time
                        remaining_time = max(0, batch_timeout - elapsed_time)
                        result = async_result.get(timeout=remaining_time)
                        results.append((problem_id, sample_id, result))
                        
                    except mp.TimeoutError:
                        print(
                            f"[WARNING] Evaluation TIMED OUT for Problem ID: {problem_id}, Sample ID: {sample_id}"
                        )
                        results.append((problem_id, sample_id, None))
                    
                        remove_cache_dir(config, problem_id, sample_id)
                    except Exception as e:
                        print(
                            f"[ERROR] Evaluation FAILED for Problem ID: {problem_id}, Sample ID: {sample_id}: {str(e)}"
                        )
                        results.append((problem_id, sample_id, None))
                        remove_cache_dir(config, problem_id, sample_id)

                end_time = time.time()

                # current batch summary
                for problem_id, sample_id, result in results:
                    print("-" * 128)
                    print(
                        f"[Eval Result] Problem ID: {problem_id}, Sample ID: {sample_id}"
                    )
                    print(result)

                    # add all the batch results here to avoid file race condition
                    # add to eval result if valid result
                    if result is not None:
                        print(f"Adding Eval Result to file for problem {problem_id} sample {sample_id}")
                        add_to_eval_results_file(problem_id, sample_id, result, eval_file_path)

                print("-" * 128)
                print(
                    f"[Curr batch] Evaluation took {end_time - start_time:.2f} seconds"
                )

                pbar.update(len(curr_work_batch))

