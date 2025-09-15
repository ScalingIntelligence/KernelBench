import os
import sys
import json
import random
from tqdm import tqdm

from src.run_utils import fetch_eval_results_for_problem
from configs import parse_autorule_args, parse_cross_model_alignment_args
from llm_utils import create_llm_client

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_DIR, "KernelBench")
AUTORULE_PATH = os.path.join(REPO_TOP_DIR, "autorule")

NUM_SAMPLES_PER_PROBLEM = 1
SAMPLE_BEST_AND_WORST = True
ALGINMENT_THRESHOLD = 0.65


def process_generated_kernels(config, run_dir):
    """
    Given directory of kernels that are evaluated, extract kernels for each problem.
    """
    # Dictionary to store processed kernels for each problem
    eval_file_path = os.path.join(run_dir, f"eval_results.json")
    with open(eval_file_path, "r") as f:
        eval_results = json.load(f)[str(config.level)]

    processed_kernels = {}

    for problem_id, prob_eval_results in eval_results.items():
        correct_kernels = []
        incorrect_kernels = []
        
        for sample_id, eval_result in prob_eval_results.items():
            if eval_result["correctness"] and eval_result["compiled"]:
                correct_kernels.append(eval_result)
            else:
                incorrect_kernels.append(eval_result)
        
        processed_kernels[problem_id] = {
            "correct": correct_kernels,
            "incorrect": incorrect_kernels
        } 
        
        # Sort correct kernels by runtime (lowest to highest)
        for problem_id in processed_kernels:
            processed_kernels[problem_id]["correct"].sort(key=lambda x: x["runtime"])
    
    # Save processed kernels to JSON file
    output_path = os.path.join(run_dir, "processed_kernels.json")
    with open(output_path, "w") as f:
        json.dump(processed_kernels, f, indent=2)
    
    print(f"Processed kernels saved to {output_path}")
    return processed_kernels


def retrieve_kernel_source_from_run_dir(run_dir, level, problem_id, sample_id):
    kernel_path = os.path.join(run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py")
    with open(kernel_path, "r") as f:
        return f.read()


def autorule(config, epoch_run_dir, llm_client):
    autorule_path = os.path.join(epoch_run_dir, "autorule")
    os.makedirs(os.path.join(autorule_path, "rule_generation"), exist_ok=True)
    os.makedirs(os.path.join(autorule_path, "rule_alignment"), exist_ok=True)
    processed_kernels = process_generated_kernels(config, os.path.join(epoch_run_dir, "generation"))

    workload = {}
    comparative_analysis_traces = {}
    for prob, data in processed_kernels.items():
        kernels = data["correct"]
        if len(kernels) < 2:
            print(f"[Comparative Analysis] Skipping Level {config.level} {prob} because it has less than 2 kernels")
            continue
        
        for sample_id in range(config.autorule_num_samples_per_problem):
            # Sample two kernels
            key = f"level{config.level}_{prob}_{sample_id}"
            if os.path.exists(os.path.join(autorule_path, "rule_generation", key, "comparative_analysis_response.json")):
                print(f"[Comparative Analysis] Skipping {key} because it already exists")
                with open(os.path.join(autorule_path, "rule_generation", key, "comparative_analysis_response.json"), "r") as f:
                    comparative_analysis_traces[key] = json.load(f)
                continue

            if config.autorule_sample_best_and_worst:
                kernel1 = kernels[0]
                kernel2 = kernels[-1]
            else:
                kernel1, kernel2 = random.sample(kernels, 2)

            kernel1_src = retrieve_kernel_source_from_run_dir(os.path.join(epoch_run_dir, "generation"), config.level, kernel1["problem_id"], kernel1["sample_id"])
            kernel2_src = retrieve_kernel_source_from_run_dir(os.path.join(epoch_run_dir, "generation"), config.level, kernel2["problem_id"], kernel2["sample_id"])
            prompt = f"""You are a kernel expert. You are given two CUDA kernels that solve the same problem. Both kernels are correct, but one is faster than the other. Analyze why one is faster than the other.
Kernel 1 (runtime: {kernel1['runtime']} ms):
```
{kernel1_src}
```

Kernel 2 (runtime: {kernel2['runtime']} ms):
```
{kernel2_src}
```
"""
            workload[key] = {"prompt": prompt, "kernel1": kernel1, "kernel2": kernel2}

  
    for key, value in tqdm(workload.items()):
        os.makedirs(os.path.join(autorule_path, "rule_generation", key), exist_ok=True)

        with open(os.path.join(autorule_path, "rule_generation", key, "comparative_analysis_prompt.txt"), "w") as f:
            f.write(value["prompt"])
        
        with open(os.path.join(autorule_path, "rule_generation", key, "comparative_analysis_kernels.json"), "w") as f:
            json.dump({"kernel1": value["kernel1"], "kernel2": value["kernel2"]}, f, indent=2)

        response = llm_client.text_completion(value["prompt"])
        reasoning = response["choices"][0]["message"]["reasoning_content"] if "reasoning_content" in response["choices"][0]["message"] else ""
        response = response["choices"][0]["message"]["content"]

        comparative_analysis_traces[key] = {"response": response, "reasoning": reasoning}
        with open(os.path.join(autorule_path, "rule_generation", key, "comparative_analysis_response.json"), "w") as f:
            json.dump({"response": response, "reasoning": reasoning}, f, indent=2)
        with open(os.path.join(autorule_path, "rule_generation", key, "comparative_analysis_response.txt"), "w") as f:
            f.write(f"REASONING:\n{reasoning}\n\nANSWER:\n{response}")


    # Step 2: Extract Rules from reasoning traces
    print("Step 2: Extract Rules from reasoning traces")
    rules = []
    for key, trace in tqdm(comparative_analysis_traces.items()):
        if os.path.exists(os.path.join(autorule_path, "rule_generation", key, "rules.json")):
            print(f"[Rules] Skipping {key} because it already exists")
            with open(os.path.join(autorule_path, "rule_generation", key, "rules.json"), "r") as f:
                rules.extend(json.load(f))
            continue

        prompt = f"""Based on the following reasoning about why one kernel is faster than the other, extract any rule-like statements implied by the reasoning to indicate the difference. Rule-like statements should be ablet to be judged objectively and determinsitcially. The rules shoud be general enough to be applied to various CUDA kernels. Below are few examples of rule-like statements:
Example 1:
- The kernel performs operator fusion between multiple operations.
Example 2:
- The kernel uses shared memory tiling to reduce global memory access.
Example 3:
- The kernel uses thread block sizes that are multiples of warp size (32).
Return the list as a JSON array of strings. Do not use ``json``, just output the JSON array directly. If there are no rule-like statements, return an empty JSON array

[Reasoning]
{trace['reasoning_trace']}
{trace['response']}
"""

        rule_response = llm_client.text_completion(prompt)
        reasoning = rule_response["choices"][0]["message"]["reasoning_content"] if "reasoning_content" in rule_response["choices"][0]["message"] else ""
        rule_response = response["choices"][0]["message"]["content"]

        with open(os.path.join(autorule_path, "rule_generation", key, "rule_response.json"), "w") as f:
            json.dump({"response": rule_response, "reasoning": reasoning}, f, indent=2)
        with open(os.path.join(autorule_path, "rule_generation", key, "rule_response.txt"), "w") as f:
            f.write(f"REASONING:\n{reasoning}\n\nANSWER:\n{rule_response}")

        try:
            if "```json" in rule_response:
                rule_response = rule_response.split("```json")[1].split("```")[0].strip()

            new_rules = json.loads(rule_response)
        except Exception as e:
            print(f"Error parsing rule response for {key}: {e}")
            try:
                if "```json" in reasoning:
                    reasoning = reasoning.split("```json")[1].split("```")[0].strip()
                new_rules = json.loads(reasoning)
            except Exception as e:
                print(f"Error parsing rule response for {key}: {e}")
                new_rules = []

        rules.extend(new_rules)

        with open(os.path.join(autorule_path, "rule_generation", key, "rules.json"), "w") as f:
            json.dump(new_rules, f, indent=2)


    # Step 3: Merge rules
    print("Step 3: Merge rules")
    if os.path.exists(os.path.join(autorule_path, "rule_generation", "merged_rules.json")):
        print(f"Skipping {config.model_name} level{config.level} merged rules because it already exists")
        return
    
    rules_str = "\n".join(rules)
    prompt = f"""Below is a large list of rule-like statements regarding the behavior of CUDA kernels. Some of these rules might be duplicates or very similar.
Please merge them so that there are no duplicates or very similar rules. Condense the rules into at most 25 rules.
Return the merged list as a JSON array of strings. Do not use ``json``, just output the JSON array directly. 
[Rules]
{rules_str}
"""
    rule_response = llm_client.text_completion(prompt, reasoning_effort="low")["choices"][0]["message"]["content"]

    if "```json" in rule_response:
        rule_response = rule_response.split("```json")[1].split("```")[0].strip()

    with open(os.path.join(autorule_path, "rule_generation", "merged_rules_response.json"), "w") as f:
        json.dump({"response": rule_response}, f, indent=2)
    with open(os.path.join(autorule_path, "rule_generation", "merged_rules_response.txt"), "w") as f:
        f.write(f"ANSWER:\n{rule_response}")

    rules = json.loads(rule_response)
    with open(os.path.join(autorule_path, "rule_generation", "merged_rules.json"), "w") as f:
        json.dump(rules, f, indent=2)

    
    # 4. Filter rules by alignment
    results = []
    for i, rule in enumerate(rules):
        print(f"Rule: {rule}")
        aligned = 0
        total = 0
        count = 0
        both_false = 0
        both_true = 0
        data = []

        rule_validation_file = os.path.join(autorule_path, "rule_alignment", f"rule_validation_rule_{i}.json")
        if os.path.exists(rule_validation_file):
            print(f"Loading results for Rule: {rule} ")
            with open(rule_validation_file, "r") as f:
                data = json.load(f)
            aligned = data["aligned"]
            total = data["total"]
            both_false = data["both_false"]
            both_true = data["both_true"]
            count = data["count"]
            data = data["data"]


        while total < config.autorule_num_alignment_samples and count < config.autorule_total_validation_limit:
            count += 1
            # Randomly sample a problem and 2 kernels
            problem = random.choice(list(processed_kernels.keys()))
            kernels = problem["correct"]
            while len(kernels) < 2:
                problem = random.choice(list(processed_kernels.keys()))
                kernels = problem["correct"]

            kernels = random.sample(kernels, 2)
            kernel1_src = retrieve_kernel_source_from_run_dir(os.path.join(epoch_run_dir, "generation"), config.level, kernels[0]["problem_id"], kernels[0]["sample_id"])
            kernel2_src = retrieve_kernel_source_from_run_dir(os.path.join(epoch_run_dir, "generation"), config.level, kernels[1]["problem_id"], kernels[1]["sample_id"])

            kernel1_is_satisfied = rule_is_satisfied(rule, kernel1_src, llm_client)
            kernel2_is_satisfied = rule_is_satisfied(rule, kernel2_src, llm_client)
            print(f"Kernel 1 is satisfied: {kernel1_is_satisfied}, Kernel 2 is satisfied: {kernel2_is_satisfied}")
            
            if kernel1_is_satisfied and kernel2_is_satisfied:
                both_true += 1
            elif not kernel1_is_satisfied and not kernel2_is_satisfied:
                both_false += 1
            elif kernel1_is_satisfied and not kernel2_is_satisfied:
                # Make sure kernel 1 is faster than kernel 2
                if kernels[0]["runtime"] < kernels[1]["runtime"]:
                    aligned += 1
                    data.append({"kernel1": kernels[0], "kernel2": kernels[1], "kernel1_is_satisfied": kernel1_is_satisfied, "kernel2_is_satisfied": kernel2_is_satisfied, "aligned": True})
                else:
                    data.append({"kernel1": kernels[0], "kernel2": kernels[1], "kernel1_is_satisfied": kernel1_is_satisfied, "kernel2_is_satisfied": kernel2_is_satisfied, "aligned": False})
                total += 1
            elif not kernel1_is_satisfied and kernel2_is_satisfied:
                if kernels[0]["runtime"] > kernels[1]["runtime"]:
                    aligned += 1
                    data.append({"kernel1": kernels[0], "kernel2": kernels[1], "kernel1_is_satisfied": kernel1_is_satisfied, "kernel2_is_satisfied": kernel2_is_satisfied, "aligned": True})
                else:
                    data.append({"kernel1": kernels[0], "kernel2": kernels[1], "kernel1_is_satisfied": kernel1_is_satisfied, "kernel2_is_satisfied": kernel2_is_satisfied, "aligned": False})
                total += 1

            
            if count % 10 == 0:
                alignment_rate = aligned / total if total > 0 else 'divide by zero'
                with open(rule_validation_file, "w") as f:
                    json.dump({
                        "rule": rule,
                        "total": total,
                        "aligned": aligned,
                        "alignment_rate": alignment_rate,
                        "both_false": both_false,
                        "both_true": both_true,
                        "count": count,
                        "data": data
                    }, f, indent=2)
        
        alignment_rate = aligned / total if total > 0 else 'divide by zero'

        with open(rule_validation_file, "w") as f:
            json.dump({
                "rule": rule,
                "total": total,
                "aligned": aligned,
                "alignment_rate": alignment_rate,
                "both_false": both_false,
                "both_true": both_true,
                "count": count,
                "data": data
            }, f, indent=2)

        print(f"Aligned: {aligned}, Total: {total}, Alignment rate: {alignment_rate}, Count: {count}")
        res = {"rule": rule, "total": total, "aligned": aligned, "alignment_rate": alignment_rate, "both_false": both_false, "both_true": both_true, "count": count}
        results.append(res)

    with open(os.path.join(autorule_path, "rule_alignment", f"rule_validation_results.json"), "w") as f:
        json.dump({"results": results}, f, indent=2)
    
    filtered_rules = [res["rule"] for res in results if res["alignment_rate"] >= config.autorule_alignment_threshold]
    with open(os.path.join(autorule_path, "rule_alignment", f"filtered_rules.json"), "w") as f:
        json.dump(filtered_rules, f, indent=2)
    with open(os.path.join(autorule_path, f"rules.json"), "w") as f:
        json.dump(filtered_rules, f, indent=2)
 
    return filtered_rules


def read_best_k_kernels(level: int, test: bool = False):
    if test:
        with open(os.path.join(KERNEL_BENCH_PATH, f"best_k_kernels_level{level}_small.json"), "r") as f:
            best_k_kernels = json.load(f)
    else:
        with open(os.path.join(KERNEL_BENCH_PATH, f"best_k_kernels_level{level}.json"), "r") as f:
            best_k_kernels = json.load(f)
    return best_k_kernels

def retrieve_kernel_source(kernel, level):
    src_file = os.path.join(REPO_TOP_DIR, "runs", kernel["run_name"], f"level_{level}_problem_{kernel['problem_id']}_sample_{kernel['sample_id']}_kernel.py")
    with open(src_file, "r") as f:
        return f.read()

def main(config):
    # Read best k kernels
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "thinking_tokens": 0}
    print(f"AutoRule framework on level {config.level} with model {config.model_name}")
    os.makedirs(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation"), exist_ok=True)

    best_k_kernels = read_best_k_kernels(config.level, test=config.test)

    # Create inference server
    llm_client = create_llm_client(os.path.join(config.run_dir, "llm_usage.json"),
                                   default_model=config.model_name,
                                   default_api_base=f"http://{config.vllm_host}:{config.vllm_port}/v1",
                                   default_temperature=config.temperature,
                                   default_max_tokens=config.max_tokens)


    # Step 1: get comparative analysis reasoning traces
    print("Step 1: get comparative analysis reasoning traces")
    
    workload = {}
    comparative_analysis_traces = {}
    for prob, kernels in best_k_kernels.items():
        if len(kernels) < 2:
            print(f"[Comparative Analysis] Skipping Level {config.level} {prob} because it has less than 2 kernels")
            continue
        
        for sample_id in range(NUM_SAMPLES_PER_PROBLEM):
            # Sample two kernels
            key = f"level{config.level}_{prob}_{sample_id}"
            if os.path.exists(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", key, "comparative_analysis_response.json")):
                print(f"[Comparative Analysis] Skipping {key} because it already exists")
                with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", key, "comparative_analysis_response.json"), "r") as f:
                    comparative_analysis_traces[key] = json.load(f)
                continue

            if SAMPLE_BEST_AND_WORST:
                kernel1 = kernels[0]
                kernel2 = kernels[-1]
            else:
                kernel1, kernel2 = random.sample(kernels, 2)

            kernel1_src = retrieve_kernel_source(kernel1, config.level)
            kernel2_src = retrieve_kernel_source(kernel2, config.level)
            prompt = f"""You are a kernel expert. You are given two CUDA kernels that solve the same problem. Both kernels are correct, but one is faster than the other. Analyze why one is faster than the other.
Kernel 1 (runtime: {kernel1['runtime']} ms):
```
{kernel1_src}
```

Kernel 2 (runtime: {kernel2['runtime']} ms):
```
{kernel2_src}
```
"""
            workload[key] = {"prompt": prompt, "kernel1": kernel1, "kernel2": kernel2}

  
    for key, value in tqdm(workload.items()):
        os.makedirs(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", key), exist_ok=True)

        with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", key, "comparative_analysis_prompt.txt"), "w") as f:
            f.write(value["prompt"])
        
        with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", key, "comparative_analysis_kernels.json"), "w") as f:
            json.dump({"kernel1": value["kernel1"], "kernel2": value["kernel2"]}, f, indent=2)

        response = llm_client.text_completion(value["prompt"])

        comparative_analysis_traces[key] = {"response": response}
        with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", key, "comparative_analysis_response.json"), "w") as f:
            json.dump({"response": response}, f, indent=2)
        with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", key, "comparative_analysis_response.txt"), "w") as f:
            f.write(f"ANSWER:\n{response}")


    # Step 2: Extract Rules from reasoning traces
    print("Step 2: Extract Rules from reasoning traces")
    rules = []
    for key, trace in tqdm(comparative_analysis_traces.items()):
        if os.path.exists(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", key, "rules.json")):
            print(f"[Rules] Skipping {key} because it already exists")
            with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", key, "rules.json"), "r") as f:
                rules.extend(json.load(f))
            continue

        prompt = f"""Based on the following reasoning about why one kernel is faster than the other, extract any rule-like statements implied by the reasoning to indicate the difference. Rule-like statements should be ablet to be judged objectively and determinsitcially. The rules shoud be general enough to be applied to various CUDA kernels. Below are few examples of rule-like statements:
Example 1:
- The kernel performs operator fusion between multiple operations.
Example 2:
- The kernel uses shared memory tiling to reduce global memory access.
Example 3:
- The kernel uses thread block sizes that are multiples of warp size (32).
Return the list as a JSON array of strings. Do not use ``json``, just output the JSON array directly. If there are no rule-like statements, return an empty JSON array

[Reasoning]
{trace['reasoning_trace']}
{trace['response']}
"""

        rule_response = llm_client.text_completion(prompt)

        with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", key, "rule_response.json"), "w") as f:
            json.dump({"response": rule_response}, f, indent=2)
        with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", key, "rule_response.txt"), "w") as f:
            f.write(f"ANSWER:\n{rule_response}")

        try:
            if "```json" in rule_response:
                rule_response = rule_response.split("```json")[1].split("```")[0].strip()

            new_rules = json.loads(rule_response)
        except Exception as e:
            print(f"Error parsing rule response for {key}: {e}")
        rules.extend(new_rules)

        with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", key, "rules.json"), "w") as f:
            json.dump(new_rules, f, indent=2)


    # Step 3: Merge rules
    print("Step 3: Merge rules")
    if os.path.exists(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", "merged_rules.json")):
        print(f"Skipping {config.model_name} level{config.level} merged rules because it already exists")
        return
    
    rules_str = "\n".join(rules)
    prompt = f"""Below is a large list of rule-like statements regarding the behavior of CUDA kernels. Some of these rules might be duplicates or very similar.
Please merge them so that there are no duplicates or very similar rules. Condense the rules into at most 25 rules.
Return the merged list as a JSON array of strings. Do not use ``json``, just output the JSON array directly. 
[Rules]
{rules_str}
"""
    rule_response = llm_client.text_completion(prompt)

    if "```json" in rule_response:
        rule_response = rule_response.split("```json")[1].split("```")[0].strip()

    with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", "merged_rules_response.json"), "w") as f:
        json.dump({"response": rule_response}, f, indent=2)
    with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", "merged_rules_response.txt"), "w") as f:
        f.write(f"ANSWER:\n{rule_response}")

    rules = json.loads(rule_response)
    with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", "merged_rules.json"), "w") as f:
        json.dump(rules, f, indent=2)
    
    print(f"Total usage AutoRule: {total_usage}")


def rule_is_satisfied(rule, kernel_src, llm_client):
    prompt = f"""You are a kernel expert. Determine whether the following CUDA kernel satisfies the following rule.
{rule}

Be as objective as possible when evaluating the rule and do not evaluate other characteristics of the response. If the rule is not applicable for this task, treat it as if the rule is satisfied. 
You must provide your answer by strictly outputting either one of the following two options:"[[Yes]]" or "[[No]]" and nothing else

Kernel:
{kernel_src}
"""
    response = llm_client.text_completion(prompt, reasoning_effort="low")
    response = response["choices"][0]["message"]["content"]
    return "Yes" in response




def rule_validation(config):
    print("#"*50)
    print("Sarting Rule Validation")
    print("#"*50)

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "thinking_tokens": 0}
    rules = json.load(open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", "merged_rules.json"), "r"))
    os.makedirs(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_validation"), exist_ok=True)
    best_kernels = read_best_k_kernels(config.level, test=config.test)

    llm_client = create_llm_client(os.path.join(config.run_dir, "llm_usage.json"),
                                   default_model=config.model_name,
                                   default_api_base=f"http://{config.vllm_host}:{config.vllm_port}/v1",
                                   default_temperature=config.temperature,
                                   default_max_tokens=config.max_tokens)

    results = []

    if config.test:
        NUM_ALIGNMENT_SAMPLES = 2
        TOTAL_VALIDATION_LIMIT = 2
    else:
        NUM_ALIGNMENT_SAMPLES = 50
        TOTAL_VALIDATION_LIMIT = 200 

    for i, rule in enumerate(rules):
        print(f"Rule: {rule}")
        aligned = 0
        total = 0
        count = 0
        both_false = 0
        both_true = 0
        data = []

        rule_validation_file = os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_validation", f"rule_validation_level_{config.level}_rule_{i}.json")
        if os.path.exists(rule_validation_file):
            print(f"Loading results for Rule: {rule} ")
            with open(rule_validation_file, "r") as f:
                data = json.load(f)
            aligned = data["aligned"]
            total = data["total"]
            both_false = data["both_false"]
            both_true = data["both_true"]
            count = data["count"]
            data = data["data"]


        while total < NUM_ALIGNMENT_SAMPLES and count < TOTAL_VALIDATION_LIMIT:
            count += 1
            # Randomly sample a problem and 2 kernels
            problem = random.choice(list(best_kernels.keys()))
            while len(best_kernels[problem]) < 2:
                problem = random.choice(list(best_kernels.keys()))

            kernels = random.sample(best_kernels[problem], 2)
            kernel1_src = retrieve_kernel_source(kernels[0], config.level)
            kernel2_src = retrieve_kernel_source(kernels[1], config.level)

            kernel1_is_satisfied = rule_is_satisfied(rule, kernel1_src, llm_client)
            kernel2_is_satisfied = rule_is_satisfied(rule, kernel2_src, llm_client)

            print(f"Kernel 1 is satisfied: {kernel1_is_satisfied}, Kernel 2 is satisfied: {kernel2_is_satisfied}")
            
            if kernel1_is_satisfied and kernel2_is_satisfied:
                both_true += 1
            elif not kernel1_is_satisfied and not kernel2_is_satisfied:
                both_false += 1
            elif kernel1_is_satisfied and not kernel2_is_satisfied:
                # Make sure kernel 1 is faster than kernel 2
                if kernels[0]["runtime"] < kernels[1]["runtime"]:
                    aligned += 1
                    data.append({"kernel1": kernels[0], "kernel2": kernels[1], "kernel1_is_satisfied": kernel1_is_satisfied, "kernel2_is_satisfied": kernel2_is_satisfied, "aligned": True})
                else:
                    data.append({"kernel1": kernels[0], "kernel2": kernels[1], "kernel1_is_satisfied": kernel1_is_satisfied, "kernel2_is_satisfied": kernel2_is_satisfied, "aligned": False})
                total += 1
            elif not kernel1_is_satisfied and kernel2_is_satisfied:
                if kernels[0]["runtime"] > kernels[1]["runtime"]:
                    aligned += 1
                    data.append({"kernel1": kernels[0], "kernel2": kernels[1], "kernel1_is_satisfied": kernel1_is_satisfied, "kernel2_is_satisfied": kernel2_is_satisfied, "aligned": True})
                else:
                    data.append({"kernel1": kernels[0], "kernel2": kernels[1], "kernel1_is_satisfied": kernel1_is_satisfied, "kernel2_is_satisfied": kernel2_is_satisfied, "aligned": False})
                total += 1

            
            if count % 10 == 0:
                alignment_rate = aligned / total if total > 0 else 'divide by zero'
                with open(rule_validation_file, "w") as f:
                    json.dump({
                        "rule": rule,
                        "total": total,
                        "aligned": aligned,
                        "alignment_rate": alignment_rate,
                        "both_false": both_false,
                        "both_true": both_true,
                        "count": count,
                        "data": data
                    }, f, indent=2)
        
        alignment_rate = aligned / total if total > 0 else 'divide by zero'

        with open(rule_validation_file, "w") as f:
            json.dump({
                "rule": rule,
                "total": total,
                "aligned": aligned,
                "alignment_rate": alignment_rate,
                "both_false": both_false,
                "both_true": both_true,
                "count": count,
                "data": data
            }, f, indent=2)

        print(f"Aligned: {aligned}, Total: {total}, Alignment rate: {alignment_rate}, Count: {count}")
        res = {"rule": rule, "total": total, "aligned": aligned, "alignment_rate": alignment_rate, "both_false": both_false, "both_true": both_true, "count": count}
        results.append(res)

       
        print(f"Total usage so far: {total_usage}")
     

    with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_validation", f"rule_validation_results_level{config.level}.json"), "w") as f:
        json.dump({"results": results, "total_usage": total_usage}, f, indent=2)

    print(f"Total usage for rule validation: {total_usage}")

def fix_rule_validation_format(config):
    rules = json.load(open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_generation", "merged_rules.json"), "r"))
    best_kernels = read_best_k_kernels(config.level, test=config.test)

    results = []


    for i, rule in enumerate(rules):
        print(f"Rule: {rule}")

        rule_validation_file = os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_validation", f"rule_validation_level_{config.level}_rule_{i}.json")
        if os.path.exists(rule_validation_file):
            print(f"Loading results for Rule: {rule} ")
            with open(rule_validation_file, "r") as f:
                contents = json.load(f)

            if "data" in contents:
                new_contents = {
                    "rule": rule,
                    "total": contents["result"]["total"],
                    "aligned": contents["result"]["aligned"],
                    "alignment_rate": contents["result"]["alignment_rate"],
                    "both_false": 0,
                    "both_true": 0,
                    "count": 150,
                    "data": contents["data"]
                }
            else:
                new_contents = {
                    "rule": rule,
                    "total": contents["result"]["total"],
                    "aligned": contents["result"]["aligned"],
                    "alignment_rate": contents["result"]["alignment_rate"],
                    "both_false": 0,
                    "both_true": 0,
                    "count": 150,
                    "data": contents
                }
            
            with open(rule_validation_file, "w") as f:
                json.dump(new_contents, f, indent=2)

            
def filter_rules(config):
    with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_validation", f"rule_validation_results_level{config.level}.json"), "r") as f:
        results = json.load(f)
    
    filtered_rules = []
    for result in results["results"]:
        if result["alignment_rate"] >= ALGINMENT_THRESHOLD:
            filtered_rules.append(result["rule"])
    
    with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "final_rules.json"), "w") as f:
        json.dump(filtered_rules, f, indent=2)


def cross_model_alignment(config):
    os.makedirs(os.path.join(AUTORULE_PATH, "cross_model_alignment"), exist_ok=True)

    llm_client_1 = create_llm_client(os.path.join(config.run_dir, "llm_usage.json"),
                                   default_model=config.model_name_1,
                                   default_api_base=f"http://{config.vllm_host_1}:{config.vllm_port_1}/v1",
                                   default_temperature=config.temperature,
                                   default_max_tokens=config.max_tokens)
    llm_client_2 = create_llm_client(os.path.join(config.run_dir, "llm_usage.json"),
                                   default_model=config.model_name_2,
                                   default_api_base=f"http://{config.vllm_host_2}:{config.vllm_port_2}/v1",
                                   default_temperature=config.temperature,
                                   default_max_tokens=config.max_tokens)    

    rules = json.load(open(config.rule_path, "r"))

    best_kernels = read_best_k_kernels(config.level, test=False)
    result = []
    for rule in rules:
        print(f"Rule: {rule}")
        total = 0
        both_true = 0
        both_false = 0

        for _ in range(100):
            # Sample one kernel
            problem = random.choice(list(best_kernels.keys()))
            while len(best_kernels[problem]) < 1:
                problem = random.choice(list(best_kernels.keys()))
        
            kernel = random.choice(best_kernels[problem])
            kernel_src = retrieve_kernel_source(kernel, config.level)
            kernel_is_satisfied_1 = rule_is_satisfied(rule, kernel_src, llm_client_1)
            kernel_is_satisfied_2 = rule_is_satisfied(rule, kernel_src, llm_client_2)

            if kernel_is_satisfied_1 and kernel_is_satisfied_2:
                both_true += 1
            elif not kernel_is_satisfied_1 and not kernel_is_satisfied_2:
                both_false += 1
    
            total += 1
        
        aligned = both_true + both_false
        
        alignment_rate = aligned / total if total > 0 else 'divide by zero'
        print(f"Alignment rate: {alignment_rate}")

        result.append({"rule": rule, "alignment_rate": alignment_rate, "total": total, "aligned": aligned, "both_true": both_true, "both_false": both_false})
    
    with open(os.path.join(AUTORULE_PATH, "cross_model_alignment", f"cross_model_alignment_results_level{config.level}_{config.model_name_1}_{config.model_name_2}.json"), "w") as f:
        json.dump(result, f, indent=2)


def autorule_main():
    args = parse_autorule_args()
    main(args)
    rule_validation(args)
    filter_rules(args)

def cross_model_alignment_main():
    args = parse_cross_model_alignment_args()
    cross_model_alignment(args)

if __name__ == "__main__":
    # autorule_main()
    cross_model_alignment_main()

