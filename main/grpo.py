import torch
import os
import yaml
import multiprocessing as mp
import wandb
import pandas as pd

import verifiers as vf
from verifiers.envs import SingleTurnEnv, MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

from utils import WorkArgs, fetch_ref_arch_from_level_problem_id
from configs import parse_args
from prompts import prompt_base
from evaluation_utils import evaluate_single_sample

from src.dataset import construct_kernelbench_dataset
from src.utils import set_gpu_arch, create_inference_server_from_presets


def get_train_dataset():
    return construct_kernelbench_dataset(1) # for now use level 1 for training

def get_eval_dataset():
    return construct_kernelbench_dataset(2) # for now use level 2 for evaluation
 

def construct_dataset(config, train=True):
    if train:
        dataset = get_train_dataset()
    else:
        dataset = get_eval_dataset()

    qa_dataset = []
    for (level, problem) in dataset:
        ref_arch_src, _ = fetch_ref_arch_from_level_problem_id(level, problem, config.dataset_src)
        question = prompt_base(ref_arch_src)
        answer = ref_arch_src
        qa_dataset.append((question, answer))
    
    df = pd.DataFrame(qa_dataset, columns=["question", "answer"])
    return df



def train(config, vf_env):
    model, tokenizer = vf.get_model_and_tokenizer(config.model_name)
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=vf.grpo_defaults(run_name=config.run_name)
    )
    trainer.train()


def eval(config, vf_env):
    inference_server = create_inference_server_from_presets(server_type="huggingface", model_name=config.model_name, max_tokens=config.max_tokens, temperature=config.temperature, num_workers=config.num_workers, api_query_interval=config.api_query_interval)
    results = vf_env.evaluate(inference_server, config.model_name, num_samples=config.num_samples)
    print(results)
    return results


def main(config):
    # Set up wandb
    tags = ["rl_training"] + config._tags.split(",")
    tags.extend([config.run_name, config.model_name])
    wandb.init(
        project="KernelBench",
        entity="j1mk1m",
        tags=tags
    )
    print(f"Starting RL training with config: {config}")

    # GPU setup 
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. Evaluation requires GPU.")
 
    set_gpu_arch(config.gpu_arch)
    assert config.num_gpu_devices <= torch.cuda.device_count(), f"Number of GPUs requested ({config.num_gpu_devices}) is greater than the number of available GPUs ({torch.cuda.device_count()})"

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    # Construct dataset
    dataset = construct_dataset(config)

    # Set up run directory
    run_dir = os.path.join(config.runs_dir, config.run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(config), f)
 
    # Evaluation
    def reward_func(prompt, completion, answer, **kwargs):
        exec_result = evaluate_single_sample(
            work_args=WorkArgs(level=config.level, problem_id=1, sample_id=0),
            configs=config,
            run_dir=run_dir
        )
        return 1.0

    kernel_rubric = Rubric(funcs=[reward_func], weights=[1.0]) 
    vf_env = vf.SingleTurnEnv(dataset=dataset, system_prompt="You are a kernel expert", rubric=kernel_rubric)

    # TODO: add multi-turn env
    class KernelMultiTurnEnv(MultiTurnEnv):
        def __init__(self, dataset, max_turns):
            rubric = Rubric(funcs=[reward_func], weights=[1.0])
            system_prompt = "You are a kernel expert"
            super().__init__(dataset=dataset, system_prompt=system_prompt, rubric=rubric, max_turns=max_turns)
        
        def env_response(self, messages, state, **kwargs):
            # eval logic to parse response and run kernel
            pass

        def is_completed(self, messages, state, **kwargs):
            return state.get("completed", False) or state.get("attempts", 0) >= self.max_turns
        
        def score_rollout(self, rollout, **kwargs):
            pass

    train(config, vf_env)


        

if __name__ == "__main__":
    configs = parse_args(rl_training=True)
    main(configs)