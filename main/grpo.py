import torch
import torch.nn as nn
import torch.optim as optim
import os
import yaml
import multiprocessing as mp
import wandb
from argparse import ArgumentParser, parse_args

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

from src.dataset import construct_kernelbench_dataset
from src.utils import set_gpu_arch
from utils import WorkArgs
from generation_utils import batch_generate
from evaluation_utils import batch_eval


def get_rewards(work_args, eval_file, config):
    # TODO
    return 1.0

def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer



def create_inference_server_from_hf_model(model, tokenizer, config):
    def call_model(inputs):
        return model.generate(inputs, max_new_tokens=config.max_tokens)



def train(config, dataset, problem_id_range, run_dir):
    eval_file = os.path.join(run_dir, "eval_results.json")

    model, tokenizer = get_model_and_tokenizer(config.model_name)
    inference_server = create_inference_server_from_hf_model(model, tokenizer)
    for i in range(config.num_train_steps):
        # Define work load
        work_args = []
        for pid in range(problem_id_range.start, problem_id_range.stop + 1):
            work_args.append(WorkArgs(problem_id=pid, sample_id=i))

        # generate
        logits = batch_generate(work_args, config, dataset, inference_server, run_dir)

        # evaluate
        batch_eval(work_args, config, dataset, inference_server, run_dir)
        rewards = get_rewards(work_args, eval_file, config)

        # gradient step
        # TODO





def main():
    argparser = ArgumentParser()
    argparser.add_argument("--run_name", type=str, default="grpo_run")
    argparser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    argparser.add_argument("--num_train_steps", type=int, default=100)
    argparser.add_argument("--gpu_arch", type=str, default="Ampere")
    argparser.add_argument("--num_gpu_devices", type=int, default=1)
    argparser.add_argument("--num_cpu_workers", type=int, default=1)
    # TODO: add args
    args = parse_args()
    args.gpu_arch = args.gpu_arch.split(",")

    tags = args.run_name.split(",")
    tags.extend([args.model_name])
    wandb.init(
        project="KernelBench",
        entity="j1mk1m",
        tags=tags
    )
    config = args
    print(f"Starting Test-Time Scaling with config: {config}")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. Evaluation requires GPU.")

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    # 1. Set up
    # Set up dataset
    # TODO: train and eval set
    train_set = construct_kernelbench_dataset(1)
    eval_set = construct_kernelbench_dataset(2)

    run_dir = os.path.join(config.runs_dir, config.run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(config), f)
 
    # set GPU arch to configure what target to build for
    set_gpu_arch(config.gpu_arch)
    assert config.num_gpu_devices <= torch.cuda.device_count(), f"Number of GPUs requested ({config.num_gpu_devices}) is greater than the number of available GPUs ({torch.cuda.device_count()})"

    train(args, train_set, eval_set, run_dir)





















# import verifiers as vf
# from verifiers.envs import SingleTurnEnv, MultiTurnEnv
# from verifiers.parsers import XMLParser
# from verifiers.rubrics import Rubric

# # Evaluation
# def reward_func(prompt, completion, answer, **kwargs):
#     return 1.0

# kernel_rubric = Rubric(funcs=[reward_func], weights=[1.0])
    

# class KernelSingleTurnEnv(SingleTurnEnv):
#     def __init__(self, dataset):
#         rubric = Rubric(funcs=[reward_func], weights=[1.0])
#         system_prompt = "You are a kernel expert"
#         super().__init__(dataset=dataset, system_prompt=system_prompt, rubric=rubric)
    
# class KernelMultiTurnEnv(MultiTurnEnv):
#     def __init__(self, dataset, max_turns):
#         rubric = Rubric(funcs=[reward_func], weights=[1.0])
#         system_prompt = "You are a kernel expert"
#         super().__init__(dataset=dataset, system_prompt=system_prompt, rubric=rubric, max_turns=max_turns)
    
#     def env_response(self, messages, state, **kwargs):
#         # eval logic to parse response and run kernel
#         pass

#     def is_completed(self, messages, state, **kwargs):
#         return state.get("completed", False) or state.get("attempts", 0) >= self.max_turns
    
#     def score_rollout(self, rollout, **kwargs):
#         pass
    


# # train.py

# model, tokenizer = vf.get_model_and_tokenizer(model_name)
# trainer = vf.GRPOTrainer(
#     model=model,
#     processing_class=tokenizer,
#     env=vf_env,
#     args=vf.grpo_defaults(run_name="...")
# )
# trainer.train()