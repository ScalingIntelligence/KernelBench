import torch
import time
import os
import yaml
import wandb
import pandas as pd
import json
from datasets import Dataset

import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
import verifiers as vf

from src.prompt_constructor import prompt_base, exec_result_to_exeution_feedback
from src.dataset import fetch_ref_arch_from_level_problem_id, TRAIN_PROBLEM_IDS_LEVEL_1, TRAIN_PROBLEM_IDS_LEVEL_2, check_in_train_dataset
from src.run_utils import find_highest_sample_id, fetch_baseline_results, write_kernel_to_disk
from src.eval import check_metadata_serializable_all_types
from src.utils import set_gpu_arch, extract_last_code

from main.configs import parse_rl_training_args, RUNS_DIR
from main.evaluation_utils import evaluate_single_sample_in_separate_process, EvaluationWorkArgs, evaluate_single_sample


BATCH_SIZE = 8


def get_train_dataset():
    return [(1, problem) for problem in TRAIN_PROBLEM_IDS_LEVEL_1] + [(2, problem) for problem in TRAIN_PROBLEM_IDS_LEVEL_2] # for now use level 1 for training

def get_eval_dataset():
    return [(1, problem) for problem in range(1, 101) if problem not in TRAIN_PROBLEM_IDS_LEVEL_1] + [(2, problem) for problem in range(1, 101) if problem not in TRAIN_PROBLEM_IDS_LEVEL_2] # for now use level 2 for evaluation
 

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

def write_eval_result_to_separate_file(level, problem, sample_id, exec_result, run_dir):
    eval_result_path = os.path.join(run_dir, f"level_{level}_problem_{problem}_sample_{sample_id}_eval_result.json")
    eval_result = {
        'level': level,
        'problem_id': problem,
        'sample_id': sample_id,
        'compiled': exec_result.compiled,
        'correctness': exec_result.correctness,
        'metadata': check_metadata_serializable_all_types(exec_result.metadata),
        'runtime': exec_result.runtime,
        'runtime_stats': exec_result.runtime_stats,
    }

    with open(eval_result_path, "w") as f:
        json.dump(eval_result, f)

def train(config, vf_env):
    torch.autograd.set_detect_anomaly(True)
    model, tokenizer = vf.get_model_and_tokenizer(config.model_name, model_kwargs={
        "torch_dtype": torch.bfloat16
    })

    grpo_config = vf.GRPOConfig(
        run_name=config.run_name,
        output_dir=os.path.join("/data/user_data/gyeongwk/grpo/", config.run_name, "checkpoints"),
        shuffle_dataset=True,
        learning_rate=2e-6,
        max_grad_norm=0.5,
        warmup_ratio=0.03,
        loss_type="dr_grpo",
        epsilon_high=0.28,
        beta=0.0,
        max_prompt_length=8192,
        temperature=config.temperature,
        max_completion_length=config.max_tokens,
        num_generations=8,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=1,
        num_batches_ahead=0,
        bf16_full_eval=True,
        gradient_checkpointing=True,
        report_to="wandb",
        vllm_server_host=config.vllm_host,
        vllm_server_port=config.vllm_port,
        max_concurrent_eval=config.max_concurrent_eval,
        eval_strategy="steps",
        max_steps=180,
        eval_steps=180,
        save_steps=90,
        logging_steps=1,
    )

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=grpo_config,
    )
    trainer.train()

    trainer.save_model(os.path.join(RUNS_DIR, config.run_name))

    eval_results = trainer.evaluate()
    print(eval_results)
    with open(os.path.join(RUNS_DIR, config.run_name, "rl_eval_results.json"), "w") as f:
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

    # Construct dataset
    dataset = construct_dataset(config)
    eval_dataset = construct_dataset(config, train=False)

    # Set up run directory
    run_dir = os.path.join(RUNS_DIR, config.run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(config), f)
 
    # Evaluation
    def reward_from_exec_result(level, problem, exec_result):
        if exec_result.correctness:
            try:
                baseline_results = fetch_baseline_results(level, problem, config.hardware)
                speedup = baseline_results["mean"] / exec_result.runtime
                return 0.3 + float(speedup)
            except Exception as e:
                print(f"Error fetching baseline results for level {level} problem {problem}: {e}")
                return 0.3
        else:
            return 0.0


    def reward_func(prompt, completion, answer, thread_id, **kwargs):
        assert len(prompt) == 2, "Single turn: Prompt should have system prompt and user prompt"
        prompt = prompt[1]["content"]
        level, problem = extract_metadata_from_prompt(prompt)
        if check_in_train_dataset(level, problem):
            sample_id = find_highest_sample_id(run_dir, level, problem, thread_id, BATCH_SIZE) # batch_size
        else:
            sample_id = find_highest_sample_id(run_dir, level, problem, 0, 1) # just find the next sample_id

        completion = completion[0]["content"]
        if config.log_response:
            response_path = os.path.join(run_dir, f"level_{level}_problem_{problem}_sample_{sample_id}_response.txt")
            with open(response_path, "w") as f:
                f.write(completion)
 

        kernel_src = extract_last_code(completion, ["python", "cpp"])
        kernel_name = f"level_{level}_problem_{problem}_sample_{sample_id}"

        if kernel_src is not None:
            write_kernel_to_disk(run_dir, level, problem, sample_id, kernel_src)

        if config.eval_mode == "local":
            device_id = (thread_id % config.max_concurrent_eval) + config.gpu_offset

            eval_device = torch.device(f'cuda:{device_id}')
            if config.verbose:
                print(f"Evaluating on device {eval_device} for sample {sample_id}")

            exec_result = evaluate_single_sample_in_separate_process(
                work_args=EvaluationWorkArgs(level=level, problem_id=problem, sample_id=sample_id, device=eval_device),
                configs=config,
                run_dir=run_dir,
                kernel_src=kernel_src, 
                kernel_name=kernel_name
            )
        elif config.eval_mode == "remote":
            exec_result = evaluate_single_sample(
                work_args=EvaluationWorkArgs(level=level, problem_id=problem, sample_id=sample_id, device=torch.device("cuda")),
                configs=config,
                run_dir=run_dir,
                kernel_src=kernel_src, 
                kernel_name=kernel_name
            )
        write_eval_result_to_separate_file(level, problem, sample_id, exec_result, run_dir)
        return reward_from_exec_result(level, problem, exec_result)
    
    kernel_rubric = vf.Rubric(funcs=[reward_func], weights=[1.0]) 
    vf_env = vf.SingleTurnEnv(dataset=dataset, 
                              eval_dataset=eval_dataset, 
                              system_prompt="You are a kernel expert", 
                              rubric=kernel_rubric)

    class KernelMultiTurnEnv(vf.MultiTurnEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
        def env_response(self, messages, state, thread_id, **kwargs):
            # eval logic to parse response and run kernel
            prompt = messages[1]["content"]
            level, problem = extract_metadata_from_prompt(prompt)
            sample_id = thread_id * 1000 + (len(messages) - 3) // 2 

            if config.log_prompt:
                # log entire trajectory
                prompt_path = os.path.join(run_dir, f"level_{level}_problem_{problem}_sample_{thread_id * 1000}_prompt.jsonl")
                with open(prompt_path, "w") as f:
                    for msg in messages:
                        f.write(json.dumps(msg) + "\n")

            last_response = messages[-1]["content"]
            kernel_src = extract_last_code(last_response, ["python", "cpp"])
            kernel_name = f"level_{level}_problem_{problem}_sample_{sample_id}"

            if config.log_response:
                response_path = os.path.join(run_dir, f"level_{level}_problem_{problem}_sample_{sample_id}_response.txt")
                with open(response_path, "w") as f:
                    f.write(last_response)

            if kernel_src is not None:
                write_kernel_to_disk(run_dir, level, problem, sample_id, kernel_src)
 
            assert config.eval_mode == "remote", "Only support remote eval for now"

            exec_result = evaluate_single_sample(
                work_args=EvaluationWorkArgs(level=level, problem_id=problem, sample_id=sample_id, device=torch.device("cuda")),
                configs=config,
                run_dir=run_dir,
                kernel_src=kernel_src, 
                kernel_name=kernel_name
            )
            write_eval_result_to_separate_file(level, problem, sample_id, exec_result, run_dir)
            env_msg = {"role": "user", "content": exec_result_to_exeution_feedback(exec_result)}
            state["level"] = level
            state["problem_id"] = problem
            if "exec_result" not in state:
                state["exec_result"] = []
            state["exec_result"].append(exec_result)
            return env_msg, state

        def is_completed(self, messages, state, **kwargs):
            return False # Keep going until max_turn is reached
        
    def multi_turn_reward_func(prompt, completion, answer, thread_id, state, **kwargs):
        exec_results = state["exec_result"]
        best_reward = 0.0
        for exec_result in exec_results:
            reward = reward_from_exec_result(state["level"], state["problem_id"], exec_result)
            if reward > best_reward:
                best_reward = reward
        return best_reward
 
    multi_turn_rubric = vf.Rubric(funcs=[multi_turn_reward_func], weights=[1.0]) 
    
    vf_multi_turn_env = KernelMultiTurnEnv(dataset=dataset, 
                                            eval_dataset=eval_dataset, 
                                            max_turns=8, 
                                            rubric=multi_turn_rubric, 
                                            system_prompt="You are a kernel expert")
        
    if config.multi_turn:
        train(config, vf_multi_turn_env)
    else:
        train(config, vf_env)
        

if __name__ == "__main__":
    configs = parse_rl_training_args()
    main(configs)