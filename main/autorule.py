import os
import sys
import json
import random
from tqdm import tqdm

from src.utils import create_inference_server_from_presets
from configs import parse_autorule_args

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_DIR, "KernelBench")
AUTORULE_PATH = os.path.join(REPO_TOP_DIR, "autorule")

NUM_SAMPLES_PER_PROBLEM = 2

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
    os.makedirs(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}"), exist_ok=True)

    best_k_kernels = read_best_k_kernels(config.level, test=config.test)

    # Create inference server
    inference_server = create_inference_server_from_presets(server_type=config.server_type,
                                                        server_address=f"http://{config.vllm_host}:{config.vllm_port}/v1",
                                                        model_name=config.model_name,
                                                        temperature=config.temperature,
                                                        max_tokens=config.max_tokens,
                                                        verbose=config.verbose)
 

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
            if os.path.exists(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", key, "comparative_analysis_response.json")):
                print(f"[Comparative Analysis] Skipping {key} because it already exists")
                with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", key, "comparative_analysis_response.json"), "r") as f:
                    comparative_analysis_traces[key] = json.load(f)
                continue

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
        os.makedirs(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", key), exist_ok=True)

        with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", key, "comparative_analysis_prompt.txt"), "w") as f:
            f.write(value["prompt"])
        
        with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", key, "comparative_analysis_kernels.json"), "w") as f:
            json.dump({"kernel1": value["kernel1"], "kernel2": value["kernel2"]}, f, indent=2)

        response, reasoning_trace, usage = inference_server(value["prompt"])

        comparative_analysis_traces[key] = {"response": response, "reasoning_trace": reasoning_trace, "usage": usage}
        with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", key, "comparative_analysis_response.json"), "w") as f:
            json.dump({"response": response, "reasoning_trace": reasoning_trace, "usage": usage}, f, indent=2)
        with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", key, "comparative_analysis_response.txt"), "w") as f:
            f.write(f"REASONING TRACE:\n{reasoning_trace}\n\nANSWER:\n{response}\n\nUsage:\n{usage}")

        total_usage["prompt_tokens"] += usage["prompt_tokens"] if "prompt_tokens" in usage else 0
        total_usage["completion_tokens"] += usage["completion_tokens"] if "completion_tokens" in usage else 0
        total_usage["total_tokens"] += usage["total_tokens"] if "total_tokens" in usage else 0
        total_usage["thinking_tokens"] += usage["thinking_tokens"] if "thinking_tokens" in usage else 0


    # Step 2: Extract Rules from reasoning traces
    print("Step 2: Extract Rules from reasoning traces")
    rules = []
    for key, trace in tqdm(comparative_analysis_traces.items()):
        if os.path.exists(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", key, "rules.json")):
            print(f"[Rules] Skipping {key} because it already exists")
            with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", key, "rules.json"), "r") as f:
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

        rule_response, rule_reasoning_trace, rule_usage = inference_server(prompt)
        total_usage["prompt_tokens"] += rule_usage["prompt_tokens"] if "prompt_tokens" in rule_usage else 0
        total_usage["completion_tokens"] += rule_usage["completion_tokens"] if "completion_tokens" in rule_usage else 0
        total_usage["total_tokens"] += rule_usage["total_tokens"] if "total_tokens" in rule_usage else 0
        total_usage["thinking_tokens"] += rule_usage["thinking_tokens"] if "thinking_tokens" in rule_usage else 0

        with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", key, "rule_response.json"), "w") as f:
            json.dump({"response": rule_response, "reasoning_trace": rule_reasoning_trace, "usage": rule_usage}, f, indent=2)
        with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", key, "rule_response.txt"), "w") as f:
            f.write(f"REASONING TRACE:\n{rule_reasoning_trace}\n\nANSWER:\n{rule_response}\n\nUsage:\n{rule_usage}")

        try:
            if "```json" in rule_response:
                rule_response = rule_response.split("```json")[1].split("```")[0].strip()

            new_rules = json.loads(rule_response)
        except Exception as e:
            print(f"Error parsing rule response for {key}: {e}")
            try:
                new_rules = json.loads(rule_reasoning_trace)
            except Exception as e:
                print(f"Error parsing rule reasoning trace for {key}: {e}")
                new_rules = []
        rules.extend(new_rules)

        with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", key, "rules.json"), "w") as f:
            json.dump(new_rules, f, indent=2)


    # Step 3: Merge rules
    print("Step 3: Merge rules")
    if os.path.exists(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "merged_rules.json")):
        print(f"Skipping {config.model_name} level{config.level} merged rules because it already exists")
        return

    rules_str = "\n".join(rules)
    prompt = f"""Below is a large list of rule-like statements regarding the behavior of CUDA kernels. Some of these rules might be duplicates or very similar.
Please merge them so that there are no duplicates or very similar rules. Condense the rules into at most 25 rules.
Return the merged list as a JSON array of strings. Do not use ``json``, just output the JSON array directly. 
[Rules]
{rules_str}
"""
    rule_response, rule_reasoning_trace, rule_usage = inference_server(prompt)
    total_usage["prompt_tokens"] += rule_usage["prompt_tokens"] if "prompt_tokens" in rule_usage else 0
    total_usage["completion_tokens"] += rule_usage["completion_tokens"] if "completion_tokens" in rule_usage else 0
    total_usage["total_tokens"] += rule_usage["total_tokens"] if "total_tokens" in rule_usage else 0
    total_usage["thinking_tokens"] += rule_usage["thinking_tokens"] if "thinking_tokens" in rule_usage else 0

    if "```json" in rule_response:
        rule_response = rule_response.split("```json")[1].split("```")[0].strip()

    with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "merged_rules_response.json"), "w") as f:
        json.dump({"response": rule_response, "reasoning_trace": rule_reasoning_trace, "usage": rule_usage}, f, indent=2)
    with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "merged_rules_response.txt"), "w") as f:
        f.write(f"REASONING TRACE:\n{rule_reasoning_trace}\n\nANSWER:\n{rule_response}\n\nUsage:\n{rule_usage}")

    rules = json.loads(rule_response)
    with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "merged_rules.json"), "w") as f:
        json.dump(rules, f, indent=2)
    
    print(f"Total usage AutoRule: {total_usage}")


def rule_is_satisfied(rule, kernel_src, inference_server):
    prompt = f"""You are a kernel expert. Determine whether the following CUDA kernel satisfies the following rule.
{rule}

Be as objective as possible when evaluating the rule and do not evaluate other characteristics of the response. If the rule is not applicable for this task, treat it as if the rule is satisfied. 
You must provide your answer by strictly outputting either one of the following two options:"[[Yes]]" or "[[No]]" and nothing else

Kernel:
{kernel_src}
"""
    response, reasoning_trace, usage = inference_server(prompt)
    return "Yes" in response, usage




def rule_validation(config):
    print("#"*50)
    print("Sarting Rule Validation")
    print("#"*50)

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "thinking_tokens": 0}
    rules = json.load(open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "merged_rules.json"), "r"))
    os.makedirs(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_validation"), exist_ok=True)
    best_kernels = read_best_k_kernels(config.level, test=config.test)

    inference_server = create_inference_server_from_presets(server_type=config.server_type,
                                                        server_address=f"http://{config.vllm_host}:{config.vllm_port}/v1",
                                                        model_name=config.model_name,
                                                        temperature=config.temperature,
                                                        max_tokens=config.max_tokens,
                                                        verbose=config.verbose)
    results = []

    if config.test:
        NUM_ALIGNMENT_SAMPLES = 2
        TOTAL_VALIDATION_LIMIT = 2
    else:
        NUM_ALIGNMENT_SAMPLES = 100
        TOTAL_VALIDATION_LIMIT = 150

    for i, rule in enumerate(rules):
        print(f"Rule: {rule}")
        aligned = 0
        total = 0
        count = 0
        both_false = 0
        both_true = 0
        data = []

        if os.path.exists(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_validation", f"rule_validation_level_{config.level}_rule_{i}.json")):
            print(f"Skipping {rule} because it already exists")
            with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_validation", f"rule_validation_level_{config.level}_rule_{i}.json"), "r") as f:
                data = json.load(f)
            if "data" in data:
                data = data["data"]
            results.append({"rule": rule, "aligned": sum([x["aligned"] for x in data]), "total": len(data), "alignment_rate": sum([x["aligned"] for x in data]) / len(data) if len(data) > 0 else "divide by zero"})
            continue

        while total < NUM_ALIGNMENT_SAMPLES and count < TOTAL_VALIDATION_LIMIT:
            count += 1
            # Randomly sample a problem and 2 kernels
            problem = random.choice(list(best_kernels.keys()))
            while len(best_kernels[problem]) < 2:
                problem = random.choice(list(best_kernels.keys()))

            kernels = random.sample(best_kernels[problem], 2)
            kernel1_src = retrieve_kernel_source(kernels[0], config.level)
            kernel2_src = retrieve_kernel_source(kernels[1], config.level)

            kernel1_is_satisfied, kernel1_usage = rule_is_satisfied(rule, kernel1_src, inference_server)
            kernel2_is_satisfied, kernel2_usage = rule_is_satisfied(rule, kernel2_src, inference_server)
            try:
                total_usage["prompt_tokens"] += kernel1_usage["prompt_tokens"] if "prompt_tokens" in kernel1_usage else 0 + kernel2_usage["prompt_tokens"] if "prompt_tokens" in kernel2_usage else 0
                total_usage["completion_tokens"] += kernel1_usage["completion_tokens"] if "completion_tokens" in kernel1_usage else 0 + kernel2_usage["completion_tokens"] if "completion_tokens" in kernel2_usage else 0
                total_usage["total_tokens"] += kernel1_usage["total_tokens"] if "total_tokens" in kernel1_usage else 0 + kernel2_usage["total_tokens"] if "total_tokens" in kernel2_usage else 0
                total_usage["thinking_tokens"] += kernel1_usage["thinking_tokens"] if "thinking_tokens" in kernel1_usage else 0 + kernel2_usage["thinking_tokens"] if "thinking_tokens" in kernel2_usage else 0
            except Exception as e:
                print(f"Error calculating usage for {rule}: {e}")

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

        print(f"Aligned: {aligned}, Total: {total}, Alignment rate: {aligned / total if total > 0 else 'divide by zero'}, Count: {count}")
        res = {"rule": rule, "aligned": aligned, "total": total, "alignment_rate": aligned / total if total > 0 else 'divide by zero'}
        results.append(res)

        with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_validation", f"rule_validation_level_{config.level}_rule_{i}.json"), "w") as f:
            json.dump({"rule": rule, "data": data, "result": res}, f, indent=2)
        
        print(f"Total usage so far: {total_usage}")
     

    with open(os.path.join(AUTORULE_PATH, config.model_name, f"level{config.level}", "rule_validation", f"rule_validation_results_level{config.level}.json"), "w") as f:
        json.dump({"results": results, "total_usage": total_usage}, f, indent=2)

    print(f"Total usage for rule validation: {total_usage}")


if __name__ == "__main__":
    args = parse_autorule_args()
    main(args)
    rule_validation(args)