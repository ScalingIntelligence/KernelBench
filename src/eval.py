"""
Bunch of helpful functions for evaluation
"""
import requests
import torch
import torch.nn as nn
import os
from pydantic import BaseModel

from src import utils

def fetch_kernel_from_database(run_name: str, problem_id: int, sample_id: int, server_url: str):
    """
    Intenral to us with our django database
    Return a dict with kernel hash, kernel code, problem_id
    """
    response = requests.get(
        f"{server_url}/get_kernel_by_run_problem_sample/{run_name}/{problem_id}/{sample_id}",
        json={"run_name": run_name, "problem_id": problem_id, "sample_id": sample_id},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert str(response_json["problem_id"]) == str(problem_id)
    return response_json["kernel"]

def fetch_ref_arch_from_problem_id(problem_id, problems) -> str:
    
    '''
    Fetches the reference architecture in string for a given problem_id
    '''
    if isinstance(problem_id, str):
        problem_id = int(problem_id)
    
    problem_path = problems[problem_id]

    # problem_path = os.path.join(REPO_ROOT_PATH, problem)
    if not os.path.exists(problem_path):
        raise FileNotFoundError(f"Problem file at {problem_path} does not exist.")
    
    ref_arch = utils.read_file(problem_path)
    return ref_arch


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class KernelExecResult(BaseModel):
    compiled: bool = False
    correctness: bool = False
    metadata: dict = {}
    # in us, only recorded if we decide to measure performance
    # can reformat this to be wall clock time
    torch_cpu_time: float = -1.0
    torch_gpu_time: float = -1.0
    custom_cpu_time: float = -1.0
    custom_gpu_time: float = -1.0

def load_original_model_and_inputs(model_original_src: str,
                                   context: dict) -> tuple[nn.Module, callable, callable]: 
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """

    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None
    
    try:
        exec(model_original_src, context) # expose to current namespace
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get('get_init_inputs')
    get_inputs_fn = context.get('get_inputs')
    Model = context.get('Model')
    return (Model, get_init_inputs_fn, get_inputs_fn)

def load_custom_model(model_custom_src: str,
                      context: dict) -> nn.Module:
    """
    Load class from custom NN.module pytorch code
    this is the code output by LLM with calls to custom cuda kernels
    """
    try:
        compile(model_custom_src, "<string>", "exec")
        exec(model_custom_src, context)
        # DANGER: need to delete refernece from global namespace
    except SyntaxError as e:
        print(f"Syntax Error in original code or Compilation Error {e}")
        return None

    ModelNew = context.get('ModelNew')
    return ModelNew

def _cleanup_cuda_extensions():
    """Helper function to cleanup compiled CUDA extensions"""
    # SIMON NOTE: is this necessary?
    import shutil
    torch_extensions_path = os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions")
    if os.path.exists(torch_extensions_path):
        shutil.rmtree(torch_extensions_path)

def graceful_eval_cleanup(curr_context: dict):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """    # delete ran-specific function definitions before next eval run
    del curr_context
     # Clear CUDA cache and reset GPU state
    torch.cuda.empty_cache()

    # does this help?
    torch.cuda.reset_peak_memory_stats()
    
    torch.cuda.synchronize()  # Wait for all CUDA operations to complete
    
    # _cleanup_cuda_extensions() # SIMON NOTE: is this necessary?

def eval_kernel_against_ref(original_model_src: str, 
                            custom_model_src: str, 
                            seed_num: int = 42, 
                            num_times: int = 1,
                            verbose: bool = False, 
                            measure_performance: bool = False,
                            device: torch.device = None) -> KernelExecResult:
    '''
    Evaluate the custom kernel against the original model

    num_times: run the evalutation multiple times and take the average
    '''
    # TODO: check device is busy
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    context = {}

    if verbose:
        print("[Eval] Start Evalulation!")
        print("[Eval] Loading Original Model")
    
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(original_model_src, context)
    set_seed(seed_num) # set seed for reproducible input
    init_inputs = get_init_inputs()
    init_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs]

    with torch.no_grad():
        set_seed(seed_num) # set seed for reproducible weights
        original_model = Model(*init_inputs)
        assert hasattr(original_model, 'forward')  
        if verbose: print("[Eval] Original Model Loaded")
    if verbose: print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")
    
    metadata = {} # for result metadata

    #this is where compilation happens
    try:
        os.environ['TORCH_USE_CUDA_DSA'] = "1"
        ModelNew = load_custom_model(custom_model_src, context)
        torch.cuda.synchronize() # not sure if this is too much 
    except Exception as e:
        print(f"Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}")
        # TODO: add metadata for compilation error (how to we get the compilation error message?)
        graceful_eval_cleanup(context)
        return KernelExecResult(compiled=False) # skip further steps
    
    try:
        with torch.no_grad():    
            set_seed(seed_num) # set seed for reproducible weights
            custom_model = ModelNew(*init_inputs)
            assert hasattr(custom_model, 'forward')  
            torch.cuda.synchronize()
        if verbose: print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(f"Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \nError: {e}")
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        graceful_eval_cleanup(context)
        metadata["runtime_error"] = e
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata) # skip further steps

    kernel_exec_result = None
    
    if measure_performance:
        if verbose: print("[Eval] Checking Both Correctness and Performance")
        raise NotImplementedError("Not implemented")
    else:
        if verbose: print("[Eval] Checking Correctness Only")
        try:
            kernel_exec_result = run_and_check_correctness(original_model, custom_model, get_inputs, num_times=num_times, verbose=verbose, seed=seed_num, device=device)
        except Exception as e:
            # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
            metadata["runtime_error"] = e
            kernel_exec_result = KernelExecResult(compiled=True, correctness=False, metadata=metadata)
            # print("EXCEPTION HAPPENS")

    graceful_eval_cleanup(context)
    return kernel_exec_result
    

def run_and_check_correctness(original_model_instance: nn.Module, 
                              new_model_instance: nn.Module, 
                              get_inputs_fn: callable, 
                              num_times: int,
                              verbose=False, 
                              seed=42,
                              device=None) -> KernelExecResult:
    """
    run the model and check correctness, 
    assume model already loaded and compiled (loaded and compiled in the caller)
    this is all on GPU, requiring cuda device and transfer .cuda()

    num_times: run the evalutation multiple times with (ideally) different random inputs to ensure correctness
    """
    pass_count = 0
    metadata = {}

    # Generate num_times seeds deterministically from the initial seed
    torch.manual_seed(seed)
    correctness_trial_seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_times)]

    with torch.no_grad():
        
        for trial in range(num_times):
            
            trial_seed = correctness_trial_seeds[trial]
            if verbose: print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

            set_seed(trial_seed)
            model = original_model_instance.cuda()

            set_seed(trial_seed)
            model_new = new_model_instance.cuda()

            output = model(*inputs)
            torch.cuda.synchronize()
            # ensure all GPU operations are completed before checking results

            try:
                output_new = model_new(*inputs)             
                torch.cuda.synchronize()
                if output.shape != output_new.shape:
                    metadata["correctness_issue"] = f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                    if verbose: print(f"[FAIL] trial {trial}: Output shape mismatch: Expected {output.shape}, got {output_new.shape}")
                    break # no hope, just never run further trials
                
                # check output value difference
                if not torch.allclose(output, output_new, atol=1e-03): # fail
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose: print(f"[FAIL] trial {trial}: Output mismatch")
                else: # pass 
                    pass_count += 1
                    if verbose: print(f"[PASS] trial {trial}: New Model matches Model")

            except Exception as e:
                print("EXCEPTION HAPPENS during correctness check")
                # NOTE: something to discusss
                # Error in launching kernel for ModelNew CUDA error: invalid configuration argument
                # Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
                # NOTE: count runtime CUDA kernel as compile issue for now
                torch.cuda.synchronize()
                # check for any CUDA errors that may have occurred
                if torch.cuda.is_available():
                    cuda_err = torch.cuda.get_last_error()
                    if cuda_err.value != 0:  # 0 means no error
                        metadata["runtime_error"] = f"Error in launching kernel for ModelNew {e}; CUDA error: {cuda_err}"
                        if verbose:
                            print(f"[FAIL] CUDA error detected: {cuda_err}")
                        break
                print(metadata)
                break

    if verbose: print(f"[Eval] Pass count: {pass_count}, num_times: {num_times}")

    # put all the useful info here!
    metadata["correctness_trials"] = f"({pass_count} / {num_times})"
    metadata["hardware"] = torch.cuda.get_device_name()

    if pass_count == num_times:


        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    else:
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

# if __name__ == "__main__":
#     fetch_kernel_from_database("kernelbench_prompt_v2_level_2", 1, 1, "http://localhost:9091")




# 