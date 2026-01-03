
"""
TODO: ongoing effort to catch reward hacking
by looking at the generated code format.


All checks are done during runtime 

Code validity checker 
- Make sure it is actually CUDA / DSL code (valid)

We support checking against 
- torch
- cuda
- triton
- thunderkittens


Call site:
- generation
- eval -> throw error to reject 
"""

import re


# you can configure this! we have suggestions 
DEFAULT_FORBIDDEN_GROUPS = [] # def not allowed
DEFAULT_WARNING_GROUPS = [] # might be valid but we want to warn

def check_valid_kernel_code(
        code: str, 
        backend: str, 
        precision: str, # fp16, fp32, bf16
        forbidden_groups: list[str] = DEFAULT_FORBIDDEN_GROUPS,
        warning_groups: list[str] = DEFAULT_WARNING_GROUPS) -> bool:
    """
    Check if the provided code is valid CUDA/DSL code for the given backend.
    """
    # TODO: Implement actual code validation logic
    match backend:
        case "cuda":
            # Validate CUDA-specific syntax
            pass
        case "triton":
            # Validate Triton-specific syntax
            pass
        case "thunderkittens":
            return check_thunderkittens_code(code)
        case _:
            # TO ADD MORE
            # Unknown backend
            return False
    return True

def check_valid_cuda(code: str) -> bool:
    """
    Check if the provided code is valid CUDA/DSL code.
    This is a placeholder for the actual implementation.
    """

    # TODO: Implement actual code validation logic

    # migrate from Kevin implementations

    return True

# list torch ops that are allowed
# we probably won't allow torch operations 

# decide degree on the torch code allowed
# do we all in cuda? do we want part of it to be in cuda?

def check_triton_code(code: str) -> bool:
    """
    Check if the provided code is valid Triton code.
    """
    # TODO: Implement actual code validation logic
    # detect it is triton jit 
    # 
    return True




# some notes here
# # 
# What prevents cheating?

# Enforcement happens mainly through Triton's compiler: @triton.jit functions cannot contain PyTorch operations – they'll fail to compile. This prevents high-level API usage in kernels. We add wrapper-side checks too. For testing, we rely on random inputs and subprocess execution. However, we trust the LLM-generated test harness itself—we don't statically analyze it for cheating.

# The compositional validation matters: individual kernel correctness doesn't guarantee the full pipeline works. Integration exposes shape mismatches, accumulating errors, and semantic misunderstandings that individual tests miss.

def check_thunderkittens_code(code: str) -> bool:
    """
    Check if the provided code is valid ThunderKittens code.
    
    Uses the following heuristics that the code:
    1. Contains ThunderKittens-specific namespace patterns:
       - "kittens::warp" or "kittens::warpgroup" or "::warpgroup::" or "::warp::" or "tma::" TODO: Get a big namespace list!
    2. Contains tile declarations:
       - st_{bf/fl}<...> (shared memory tiles)
       - rt_{bf/fl}<...> (register tiles)
    """
    # (1) Namespace patterns to search for: if it contains a single one of these, then it's valid!
    # TODO: For more complicated programs, you really want to make sure it's following the producer-consumer pattern
    #   - could search specifically for "producer" and "consumer" structs, or the presence of "tma::load_async"
    warp_patterns = [
        r"kittens::warp\b",
        r"kittens::warpgroup\b",
        r"::warpgroup::",
        r"::warp::",
        r"warpgroup::",
        r"warp::"
    ]
    has_warp_pattern = any(re.search(pattern, code) for pattern in warp_patterns)
    if not has_warp_pattern:
        return False
    
    # (2) Check that the file actually uses tiles: st_<type><...> or rt_<type><...>
    # Pattern matches: st_bf<...>, st_fl<...>, rt_bf<...>, rt_fl<...>, etc. (any type)
    # TODO: we don't look for global gl here...
    # Also handles namespaced versions like kittens::st_bf<...>
    tile_pattern = r"(?:kittens::)?(?:st|rt)_\w+\s*<[^>]+>"
    has_tiles = bool(re.search(tile_pattern, code))
    if not has_tiles:
        return False

    # # (3) Producer-Consumer semantics
    # pc_patterns = [
    #     r"tma::\b",
    #     r"load_async\b"
    # ]
    # has_pc_pattern = any(re.search(pattern, code) for pattern in pc_patterns)
    # if not has_pc_pattern:
    #     return False

    passes_compilation = False
    # TODO: Try compilation here and return false if it fails

    
    # ALL of the above conditions must be met for this code string to pass
    return True


### from ohter projects



# from kevin this is my reward function hacks
# we will strip part of the code to use generic checker functions
def reward_func(queries, prompts, labels, metadata, external_model=False):
    worker_id = metadata.get("rank", 0)
    global_step = metadata.get("step", 10000)
    refinement_step = metadata.get("refinement_step", 0)
    train = metadata.get("train", True)  # Get the train flag from metadata

    sample_indices = metadata.get("sample_indices", [0] * len(queries))  # Extract refinement indices with default
    print(f"[Reward {worker_id}] Refinement Step {refinement_step} Sample Indices: {sample_indices}")
    print(f"[Reward {worker_id}] Train mode: {train}")
    labels = [json.loads(label) for label in labels]
    ref_arch_src = [label["ref_arch_src"] for label in labels]
    baseline_runtime = [label["baseline_runtime"] for label in labels]
    level = [label["level"] for label in labels]
    task_id = [label["task_id"] for label in labels]
    output_dir = labels[0]["output_dir"]
    base_process_index = worker_id
    if not external_model:
        parse_pattern = r".*?</think>.*?```(.*?)```.*?$"
    else:
        parse_pattern = r"```(.*?)```"
    verbose = False

    eval_cache_dir = f"{output_dir}/eval_cache"
    os.makedirs(eval_cache_dir, exist_ok=True)

    # Initialize result storage
    rewards = [0.0] * len(queries)
    tool_answers = [""] * len(queries)
    
    def save_output_entry(entry):
        arch_output_dir = f"{output_dir}/step_{global_step}"
        arch_output_path = f"{arch_output_dir}/device_{base_process_index}.json"
        os.makedirs(arch_output_dir, exist_ok=True)
        
        # Read existing data or create new array
        data = []
        if os.path.exists(arch_output_path):
            try:
                with open(arch_output_path, 'r') as f:
                    data = json.loads(f.read())
            except json.JSONDecodeError:
                pass
        
        # Append and write back
        data.append(entry)
        with open(arch_output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def handle_parsing_error(idx, content, error_type, error_msg, reward=0.0):
        min_length, max_length = 0, 0  # Set appropriate values or make these parameters
        len_reward = calculate_length_reward(False, len(content), min_length, max_length)
        rewards[idx] = reward

        # Create an entry for the error
        eval_result = KernelExecResult()
        eval_result.metadata = register_and_format_exception(error_type, error_msg, eval_result.metadata)
        entry = _create_entry(
            level[idx], task_id[idx], global_step, base_process_index, 
            eval_result, baseline_runtime[idx], reward, prompts[idx], 
            content, min_length, max_length, error_type, error_msg, 
            len_reward, 0, refinement_step=refinement_step,
            sample_index=sample_indices[idx]  # Add refinement index
        )
        eval_result_dict = response_for_kernel_eval(eval_result, external_model)
        
        # Save the entry
        save_output_entry(entry)
        return eval_result_dict  # Parsing failed

    # Parse completions and prepare valid items for processing
    valid_items = []
    for idx, (prompt, completion, ref_arch, base_runtime, ind_level, id) in enumerate(zip(prompts, queries, ref_arch_src, baseline_runtime, level, task_id)):
        # Create a unique process index for each task
        unique_process_index = f"{base_process_index}_{idx}"
        eval_cache_path = f"{eval_cache_dir}/eval_results_{unique_process_index}.json"
        
        # Parse the completion to extract CUDA kernel
        matches = re.match(parse_pattern, completion, re.DOTALL)
        if not matches: # prompt comes with 2 codeblocks
            print(f"[Reward {unique_process_index} EXIT] had no match")
            eval_result_dict = handle_parsing_error(idx, completion, "parsing_error", "Parsing failed. No valid code block or thinking too long.", 0.0)
            tool_answers[idx] = eval_result_dict
            continue

        # Extract and clean up the CUDA kernel code
        custom_cuda = matches.group(1).strip()
        for code_type in ["python", "cpp", "cuda"]:
            if custom_cuda.startswith(code_type):
                custom_cuda = custom_cuda[len(code_type):].strip()

        # Verify the CUDA kernel looks valid
        if ("__global__" not in custom_cuda) or ("load_inline(" not in custom_cuda):
            print(f"[Reward {unique_process_index} EXIT] output has no cuda kernel")
            eval_result_dict = handle_parsing_error(idx, completion, "parsing_error", "Parsing failed. Has no valid CUDA kernel.", 0.0)
            tool_answers[idx] = eval_result_dict
            continue

        # Verify the CUDA kernel does not use torch.nn (except for allowed utility functions)
        # Split code into lines and filter out comments
        code_lines = [line for line in custom_cuda.split('\n') if not line.strip().startswith('#')]
        non_comment_code = '\n'.join(code_lines)
        
        # Check for disallowed nn patterns
        if re.search(
            r'nn\.(?!(Module|parameter|Parameter|ParameterList|ParameterDict|ModuleList|ModuleDict|init)\b)',
            non_comment_code
        ):
            print(f"[Reward {unique_process_index} EXIT] output has disallowed nn. in custom cuda")
            eval_result_dict = handle_parsing_error(idx, completion, "parsing_error", 
                "Parsing failed. You're not allowed to use torch.nn in the custom CUDA kernel (except for containers, nn.Parameter, and nn.init).", 0.0)
            tool_answers[idx] = eval_result_dict
            continue

        # List of disallowed torch functions
        disallowed_torch_functions = [
            "torch.conv1d", "torch.conv2d", "torch.conv3d",
            "torch.conv_transpose1d", "torch.conv_transpose2d", "torch.conv_transpose3d",
            "torch.avg_pool1d", "torch.avg_pool2d", "torch.avg_pool3d",
            "torch.max_pool1d", "torch.max_pool2d", "torch.max_pool3d",
            "torch.adaptive_avg_pool1d", "torch.adaptive_avg_pool2d", "torch.adaptive_avg_pool3d",
            "torch.adaptive_max_pool1d", "torch.adaptive_max_pool2d", "torch.adaptive_max_pool3d",
            "torch.relu", "torch.hardtanh", "torch.elu", "torch.selu",
            "torch.leaky_relu", "torch.gelu", "torch.softsign", "torch.softplus",
            "torch.softmax", "torch.log_softmax", "torch.tanh", "torch.sigmoid",
            "torch.hardsigmoid", "torch.silu", "torch.mish",
            "torch.batch_norm", "torch.group_norm", "torch.layer_norm",
            "torch.instance_norm", "torch.rms_norm", "torch.normalize", "torch.linear",
            "torch.cross_entropy", "torch.kl_div", "torch.mse_loss",
            "torch.huber_loss", "torch.triplet_margin_loss", "torch.cosine_similarity",
            "torch.logsumexp", "torch.log_softmax", "torch.clamp", "torch.dropout"
        ]

        # Create efficient regex pattern for all disallowed functions
        # The pattern uses a lookahead (?=...) to ensure we only match complete function references:
        # - \s*\( : followed by parentheses (function call)
        # - \s    : followed by whitespace (end of identifier)
        # - $     : at the end of a line
        disallowed_pattern = r'\b(' + '|'.join(re.escape(func) for func in disallowed_torch_functions) + r')(?=\s*\(|\s|$)'

        # Single regex search instead of multiple string checks
        match = re.search(disallowed_pattern, non_comment_code)
        if match:
            print(f"[Reward {unique_process_index} EXIT] output uses disallowed function by calling nn.Functional from torch: {match.group(0)}")
            eval_result_dict = handle_parsing_error(idx, completion, "parsing_error", 
                f"Parsing failed. You're not allowed to use torch.nn.functional in the custom CUDA kernel by calling {match.group(0)}.", 0.0)
            tool_answers[idx] = eval_result_dict
            continue
        
        # If we reach here, parsing was successful - prepare item data
        item_data = {
            "prompt": prompt,
            "completion": completion,
            "ref_arch": ref_arch,
            "baseline_runtime": base_runtime,
            "ind_level": ind_level,
            "id": id,
            "process_index": unique_process_index,
            "current_step": global_step,
            "refinement_step": refinement_step,
            "sample_index": sample_indices[idx],  # Add refinement index
            "device": base_process_index,
            "output_dir": output_dir,
            "eval_cache_path": eval_cache_path,
            "custom_cuda": custom_cuda
        }
        valid_items.append((idx, item_data))

    # Process valid items if any exist
    if valid_items:
        
        # Determine number of workers based on available items
        max_workers = min(len(valid_items), MAX_CPU)
        print(f"[Reward {worker_id}] Running with {max_workers} workers for {len(valid_items)} items")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            futures = []
            for original_idx, item in valid_items:
                future = executor.submit(_process_single_item, item, verbose, train, external_model)
                futures.append((future, original_idx))
                
            # Process results
            for future, original_idx in futures:
                try:
                    reward, eval_result_dict, entry = future.result()
                    rewards[original_idx] = reward
                    tool_answers[original_idx] = eval_result_dict
                    
                    # Save the entry using the helper function
                    save_output_entry(entry)
                        
                except Exception as exc:
                    print(f"[Reward {worker_id}_{original_idx}] Processing item generated an exception: {exc}")
    else:
        print(f"[Reward {worker_id}] No valid items to process")
        
    # Clean up temp files
    for idx, item_data in valid_items:
        try:
            if os.path.exists(item_data["eval_cache_path"]):
                os.unlink(item_data["eval_cache_path"])
        except:
            pass

    print(f"[Reward {worker_id} Refinement Step {refinement_step}] Finished processing {len(valid_items)} items")
    print(f"[Reward {worker_id} Refinement Step {refinement_step}] Rewards: {rewards}")
    
    return rewards, tool_answers


# from another project
def check_for_cheating(kernel_code: str) -> bool:
    """
    Check if the kernel code is cheating using Kevin-32B paper rules.

    Kevin paper identifies three cheating patterns that receive zero reward:
    1. PyTorch wrapping: Using torch.nn or torch.nn.functional
    2. Try-except fallback: Wrapping incomplete CUDA in exception handlers
    3. Inheritance bypass: Using 'pass' to inherit from reference

    Reference: arXiv:2507.11948 "Kevin: Multi-Turn RL for Generating CUDA Kernels"
    """
    # Kevin Rule 1: PyTorch wrapping detection
    # Zero reward for kernels containing torch.nn or torch.nn.functional
    pytorch_patterns = [
        "torch.nn.functional",
        "torch.nn.",
        "F.conv", "F.linear", "F.relu", "F.gelu", "F.softmax",
        "F.batch_norm", "F.layer_norm", "F.dropout",
        "nn.functional.",
    ]
    for pattern in pytorch_patterns:
        if pattern in kernel_code:
            return True

    # Kevin Rule 2: Try-except fallback pattern detection
    # Models wrap incomplete CUDA in exception handlers that fall back to PyTorch
    # Zero reward for kernels containing try/except
    if "try:" in kernel_code or "except:" in kernel_code or "except " in kernel_code:
        return True

    # Kevin Rule 3: Inheritance bypass detection
    # Model inherits from reference using 'pass'
    # Zero reward for kernels containing pass statements
    # Be careful to not match 'passed' or similar words
    import re
    if re.search(r'\bpass\b', kernel_code):
        return True

    # Additional check: no custom kernel implementation at all
    has_triton_kernel = "@triton.jit" in kernel_code or "@triton.autotune" in kernel_code
    has_cuda_kernel = "load_inline" in kernel_code or "cpp_extension" in kernel_code
    has_cute_kernel = "cute::" in kernel_code or "from cutlass" in kernel_code
    has_tilelang = "@T.prim_func" in kernel_code or "tvm.build" in kernel_code

    has_custom_implementation = any([
        has_triton_kernel,
        has_cuda_kernel,
        has_cute_kernel,
        has_tilelang
    ])

    # If no custom implementation, check for direct torch ops
    if not has_custom_implementation:
        torch_ops = [
            "torch.mm", "torch.bmm", "torch.matmul",
            "torch.conv", "torch.einsum",
        ]
        for op in torch_ops:
            if op in kernel_code:
                return True

    return False