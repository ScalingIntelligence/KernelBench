import os

from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import extract_first_code, read_file, maybe_multithread

from configs import TestTimeScalingConfig, WorkArgs


def generate_sample_single(work: WorkArgs, config: TestTimeScalingConfig, dataset, inference_server: callable, run_dir: str) -> bool:
    # 1. Fetch Problem
    if config.dataset_src == "huggingface":
        curr_problem_row = dataset.filter(lambda x: x["problem_id"] == work.problem_id, desc=None)

        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

    elif config.dataset_src == "local":
        problem_idx_in_dataset = work.problem_id - 1 # due to dataset list being 0-indexed locally
        ref_arch_path = dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == work.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"  

    # Construct Prompt   
    custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
    if config.log_prompt:
        prompt_path = os.path.join(run_dir, f"level_{config.level}_problem_{work.problem_id}_sample_{work.sample_id}_prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(custom_cuda_prompt)

    # Query server with constructed prompt
    custom_cuda = inference_server(custom_cuda_prompt)
    custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"

    if config.verbose:
        print(f"Generated sample {work.sample_id} for problem {problem_number}: {problem_name}")

    # Store to local file
    kernel_path = os.path.join(run_dir, f"level_{config.level}_problem_{work.problem_id}_sample_{work.sample_id}_kernel.py")
    with open(kernel_path, "w") as f:
        f.write(custom_cuda)
    
    return True

def generate_sample_launcher(work: WorkArgs, config: TestTimeScalingConfig, dataset, inference_server: callable, run_dir: str):
    try:
        return generate_sample_single(work, config, dataset, inference_server, run_dir)
    except Exception as e:
        print(f"Error generating sample {work.problem_id} {work.sample_id}: {e}")
        return None


def batch_generate(
    total_work: list[WorkArgs],
    config: TestTimeScalingConfig,
    dataset,
    inference_server: callable,
    run_dir: str,
):
    return maybe_multithread(generate_sample_launcher, 
                      total_work, 
                      config.num_workers, 
                      time_interval=config.api_query_interval, 
                      # extra args
                      config=config, 
                      dataset=dataset, 
                      inference_server=inference_server,
                      run_dir=run_dir
                      )

