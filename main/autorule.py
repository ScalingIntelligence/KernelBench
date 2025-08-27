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
    best_k_kernels_level1 = read_best_k_kernels(1, test=config.test)
    best_k_kernels_level2 = read_best_k_kernels(2, test=config.test)

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
    for prob, kernels in best_k_kernels_level1.items():
        if len(kernels) < 2:
            print(f"[Comparative Analysis] Skipping Level 1 {prob} because it has less than 2 kernels")
            continue
        
        for sample_id in range(NUM_SAMPLES_PER_PROBLEM):
            # Sample two kernels
            key = f"level1_{prob}_{sample_id}"
            if os.path.exists(os.path.join(AUTORULE_PATH, f"{key}_comparative_analysis_response.json")):
                print(f"[Comparative Analysis] Skipping {key} because it already exists")
                with open(os.path.join(AUTORULE_PATH, f"{key}_comparative_analysis_response.json"), "r") as f:
                    comparative_analysis_traces[key] = json.load(f)
                continue

            kernel1, kernel2 = random.sample(kernels, 2)
            kernel1_src = retrieve_kernel_source(kernel1, 1)
            kernel2_src = retrieve_kernel_source(kernel2, 1)
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

    for prob, kernels in best_k_kernels_level2.items():
        if len(kernels) < 2:
            print(f"[Comparative Analysis] Skipping Level 2 {prob} because it has less than 2 kernels")
            continue
        
        for sample_id in range(NUM_SAMPLES_PER_PROBLEM):
            # Sample two kernels
            key = f"level2_{prob}_{sample_id}"
            if os.path.exists(os.path.join(AUTORULE_PATH, f"{key}_comparative_analysis_response.json")):
                print(f"[Comparative Analysis] Skipping {key} because it already exists")
                with open(os.path.join(AUTORULE_PATH, f"{key}_comparative_analysis_response.json"), "r") as f:
                    comparative_analysis_traces[key] = json.load(f)
                continue

            kernel1, kernel2 = random.sample(kernels, 2)
            kernel1_src = retrieve_kernel_source(kernel1, 2)
            kernel2_src = retrieve_kernel_source(kernel2, 2)
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
        with open(os.path.join(AUTORULE_PATH, f"{key}_comparative_analysis_prompt.txt"), "w") as f:
            f.write(value["prompt"])
        
        with open(os.path.join(AUTORULE_PATH, f"{key}_comparative_analysis_kernels.json"), "w") as f:
            json.dump({"kernel1": value["kernel1"], "kernel2": value["kernel2"]}, f, indent=2)

        response, reasoning_trace, usage = inference_server(value["prompt"])
        comparative_analysis_traces[key] = {"response": response, "reasoning_trace": reasoning_trace, "usage": usage}
        with open(os.path.join(AUTORULE_PATH, f"{key}_comparative_analysis_response.json"), "w") as f:
            json.dump({"response": response, "reasoning_trace": reasoning_trace, "usage": usage}, f, indent=2)
        with open(os.path.join(AUTORULE_PATH, f"{key}_comparative_analysis_response.txt"), "w") as f:
            f.write(f"REASONING TRACE:\n{reasoning_trace}\n\nANSWER:\n{response}\n\nUsage:\n{usage}")


    # Step 2: Extract Rules from reasoning traces
    print("Step 2: Extract Rules from reasoning traces")
    rules = []
    for key, trace in tqdm(comparative_analysis_traces.items()):
        if os.path.exists(os.path.join(AUTORULE_PATH, f"{key}_rules.json")):
            print(f"[Rules] Skipping {key} because it already exists")
            with open(os.path.join(AUTORULE_PATH, f"{key}_rules.json"), "r") as f:
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
        with open(os.path.join(AUTORULE_PATH, f"{key}_rule_response.json"), "w") as f:
            json.dump({"response": rule_response, "reasoning_trace": rule_reasoning_trace, "usage": rule_usage}, f, indent=2)
        with open(os.path.join(AUTORULE_PATH, f"{key}_rule_response.txt"), "w") as f:
            f.write(f"REASONING TRACE:\n{rule_reasoning_trace}\n\nANSWER:\n{rule_response}\n\nUsage:\n{rule_usage}")

        try:
            new_rules = json.loads(rule_response)
        except Exception as e:
            print(f"Error parsing rule response for {key}: {e}")
            try:
                new_rules = json.loads(rule_reasoning_trace)
            except Exception as e:
                print(f"Error parsing rule reasoning trace for {key}: {e}")
                new_rules = []
        rules.extend(new_rules)

        with open(os.path.join(AUTORULE_PATH, f"{key}_rules.json"), "w") as f:
            json.dump(new_rules, f, indent=2)


    # Step 3: Merge rules
    print("Step 3: Merge rules")
    rules_str = "\n".join(rules)
    prompt = f"""Below is a large list of rule-like statements regarding the behavior of CUDA kernels. Some of these rules might be duplicates or very similar.
Please merge them so that there are no duplicates or very similar rules.
Return the merged list as a JSON array of strings. Do not use ``json``, just output the JSON array directly. 
[Rules]
{rules_str}
"""
    rule_response, rule_reasoning_trace, rule_usage = inference_server(prompt)
    with open(os.path.join(AUTORULE_PATH, "merged_rules_response.json"), "w") as f:
        json.dump({"response": rule_response, "reasoning_trace": rule_reasoning_trace, "usage": rule_usage}, f, indent=2)
    with open(os.path.join(AUTORULE_PATH, "merged_rules_response.txt"), "w") as f:
        f.write(f"REASONING TRACE:\n{rule_reasoning_trace}\n\nANSWER:\n{rule_response}\n\nUsage:\n{rule_usage}")

    rules = json.loads(rule_response)
    with open(os.path.join(AUTORULE_PATH, "merged_rules.json"), "w") as f:
        json.dump(rules, f, indent=2)



if __name__ == "__main__":
    args = parse_autorule_args()
    main(args)