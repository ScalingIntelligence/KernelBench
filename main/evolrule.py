import torch
import multiprocessing as mp
import os
import yaml
from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.utils import set_gpu_arch, create_inference_server_from_presets

from main.test_time_scaling import iterative_refinement
from main.configs import parse_evolrule_args, RUNS_DIR
from main.autorule import autorule



def main(config):
    """
    Test-Time Scaling for Particular Level
    """
    print(f"Starting EvolRule with config: {config}")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. Evaluation requires GPU.")

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    # 1. Set up
    # Set up dataset
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    num_problems_in_level = len(curr_level_dataset)

    if config.subset == (None, None):
        problem_id_range = range(1, num_problems_in_level)
    else:
        assert config.subset[0] >= 1 and config.subset[1] <= num_problems_in_level, f"Subset range {config.subset} out of range for Level {config.level}"
        problem_id_range = range(config.subset[0], config.subset[1])

    print(f"Level {config.level} problems: {problem_id_range}")

    # set up run directory
    run_dir = os.path.join(RUNS_DIR, config.run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(config), f)
 
    assert config.store_type == "local", "supporting local file-system based storage for now" # database integreation coming soon, need to migrate from CUDA Monkeys code

    # set GPU arch to configure what target to build for
    set_gpu_arch(config.gpu_arch)
    assert config.num_eval_devices <= torch.cuda.device_count(), f"Number of GPUs requested ({config.num_eval_devices}) is greater than the number of available GPUs ({torch.cuda.device_count()})"

    # Create inference function with config parameters
    inference_server = create_inference_server_from_presets(server_type=config.server_type,
                                                        server_address=f"http://{config.vllm_host}:{config.vllm_port}/v1",
                                                        model_name=config.model_name,
                                                        temperature=config.temperature,
                                                        max_tokens=config.max_tokens,
                                                        verbose=config.verbose)
    
    rule_path = None
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch}")
        epoch_run_dir = os.path.join(run_dir, f"epoch_{epoch}")
        os.makedirs(epoch_run_dir, exist_ok=True)

        # 1. Generation
        generation_dir = os.path.join(epoch_run_dir, "generation")
        iterative_refinement(config, config.level, problem_id_range, inference_server, generation_dir, rule_path)


        # 2. AutoRule: Comparative Analysis
        rules = autorule(config, epoch_run_dir, inference_server)
        print(f"Rules: {rules}")

        # 3. Prompt Evolution (update rule_path)
        rule_path = os.path.join(epoch_run_dir, "autorule", "rules.json")
    

if __name__ == "__main__":
    args = parse_evolrule_args()
    main(args)
