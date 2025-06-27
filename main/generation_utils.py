import os
import traceback

from src.utils import maybe_multithread, extract_last_code

from utils import WorkArgs
from prompts import generate_prompt
from dataset import fetch_ref_arch_from_level_problem_id
from run_utils import check_if_kernel_exists


def generate_sample_single(work: WorkArgs, config, inference_server: callable, run_dir: str) -> bool:
    ref_arch_src, _ = fetch_ref_arch_from_level_problem_id(work.level, work.problem_id, config.dataset_src)

    # Construct Prompt   
    custom_cuda_prompt = generate_prompt(work, config, ref_arch_src, inference_server, run_dir)
    if config.log_prompt:
        prompt_path = os.path.join(run_dir, f"level_{work.level}_problem_{work.problem_id}_sample_{work.sample_id}_prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(custom_cuda_prompt)

    # Query server with constructed prompt
    custom_cuda, reasoning_trace, usage = inference_server(custom_cuda_prompt)
    if config.log_response:
        response_path = os.path.join(run_dir, f"level_{work.level}_problem_{work.problem_id}_sample_{work.sample_id}_response.txt")
        response = f"REASONING TRACE:\n{reasoning_trace}\n\nANSWER:\n{custom_cuda}\n\nUsage:\n{usage}"
        with open(response_path, "w") as f:
            f.write(response)
    custom_cuda = extract_last_code(custom_cuda, ["python", "cpp"])

    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"

    if config.verbose:
        print(f"Generated sample {work.sample_id} for problem {work.problem_id}")

    # Store to local file
    kernel_path = os.path.join(run_dir, f"level_{work.level}_problem_{work.problem_id}_sample_{work.sample_id}_kernel.py")
    with open(kernel_path, "w") as f:
        f.write(custom_cuda)
    
    return True

def generate_sample_launcher(work: WorkArgs, config, inference_server: callable, run_dir: str):
    try:
        return generate_sample_single(work, config, inference_server, run_dir)
    except Exception as e:
        print(f"Error generating problem {work.problem_id} sample {work.sample_id}: {e}")
        print(traceback.format_exc()) 
        return None


def batch_generate(
    total_work: list[WorkArgs],
    config,
    inference_server: callable,
    run_dir: str,
):
    total_work = [work for work in total_work if not check_if_kernel_exists(run_dir, work.level, work.problem_id, work.sample_id)]
    return maybe_multithread(generate_sample_launcher, 
                      total_work, 
                      config.num_workers, 
                      pbar_name=f"Generation {config.method} Progress",
                      time_interval=config.api_query_interval, 
                      # extra args
                      config=config, 
                      inference_server=inference_server,
                      run_dir=run_dir
                      )

