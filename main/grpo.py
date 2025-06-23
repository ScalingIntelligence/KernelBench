import torch
import os
import yaml
import multiprocessing as mp
import wandb
import pandas as pd
import json
from datasets import Dataset

import verifiers as vf

from configs import parse_args
from prompts import prompt_base
from evaluation_utils import evaluate_single_sample, EvaluationWorkArgs
from dataset import construct_kernelbench_dataset, fetch_ref_arch_from_level_problem_id
from run_manager import find_highest_sample_id, fetch_baseline_results, write_kernel_to_disk

from src.utils import set_gpu_arch


def get_train_dataset():
    return [(1, problem) for problem in range(1, 11)] # for now use level 1 for training

def get_eval_dataset():
    return [(2, problem) for problem in range(1, 11)] # for now use level 2 for evaluation
 

def construct_dataset(config, train=True):
    if train:
        dataset = get_train_dataset()
    else:
        dataset = get_eval_dataset()

    qa_dataset = []
    for (level, problem) in dataset:
        ref_arch_src, _ = fetch_ref_arch_from_level_problem_id(level, problem, config.dataset_src)
        question = f"Level {level} Problem {problem}:\n" + prompt_base(ref_arch_src)
        answer = ref_arch_src
        qa_dataset.append((question, answer))
    
    df = Dataset.from_pandas(pd.DataFrame(qa_dataset, columns=["question", "answer"]))
    return df

def extract_metadata_from_prompt(prompt):
    level = int(prompt.split("Level ")[1].split("Problem ")[0].strip())
    problem = int(prompt.split("Problem ")[1].split(":")[0].strip())
    return level, problem



def train(config, vf_env):
    model, tokenizer = vf.get_model_and_tokenizer(config.model_name)

    grpo_config = vf.GRPOConfig(
        run_name=config.run_name,
        output_dir=os.path.join(config.runs_dir, config.run_name, "checkpoints"),
        learning_rate=1e-5,
        max_prompt_length=None,
        eval_steps=100,
        save_steps=50,
        logging_steps=10,
        gradient_checkpointing=True,
        report_to="wandb"
    )
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=grpo_config,
        eval_dataset=construct_dataset(config, train=False)
    )
    trainer.train()

    trainer.save_model(os.path.join(config.runs_dir, config.run_name))

    eval_results = trainer.evaluate()
    print(eval_results)
    with open(os.path.join(config.runs_dir, config.run_name, "rl_eval_results.json"), "w") as f:
        json.dump(eval_results, f)


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
    def reward_from_exec_result(level, problem, exec_result):
        if exec_result.is_correct:
            baseline_results = fetch_baseline_results(level, problem, config.hardware)
            speedup = baseline_results["mean"] / exec_result.runtime
            return 0.3 + speedup
        else:
            return 0.0


    def reward_func(prompt, completion, answer, **kwargs):
        level, problem = extract_metadata_from_prompt(prompt)
        sample_id = find_highest_sample_id(level, problem, run_dir) + 1

        write_kernel_to_disk(run_dir, level, problem, sample_id, completion)

        exec_result = evaluate_single_sample(
            work_args=EvaluationWorkArgs(level=level, problem_id=problem, sample_id=sample_id, device=config.eval_device),
            configs=config,
            run_dir=run_dir
        )
        return reward_from_exec_result(level, problem, exec_result)

    kernel_rubric = vf.Rubric(funcs=[reward_func], weights=[1.0]) 
    vf_env = vf.SingleTurnEnv(dataset=dataset, system_prompt="You are a kernel expert", rubric=kernel_rubric)

    # TODO: add multi-turn env
    class KernelMultiTurnEnv(vf.MultiTurnEnv):
        def __init__(self, dataset, max_turns):
            rubric = kernel_rubric
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