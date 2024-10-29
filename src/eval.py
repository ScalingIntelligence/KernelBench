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
    metadata: str = ""
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

    exec(model_original_src, context) # expose to global namespace

    # these should be defined in the model
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

def eval_kernel_against_ref(original_model_src: str, 
                            custom_model_src: str, 
                            seed_num: int =42, 
                            num_times: int =1,
                            verbose: bool =False, 
                            measure_performance: bool =False,
                            device: torch.device =None) -> KernelExecResult:
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
        if verbose:
            print("[Eval] Original Model Loaded")
    
    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")
    #this is where compilation happens
    try:
        ModelNew = load_custom_model(custom_model_src, context)
    except Exception as e:
        print(f"Failed to compile custom CUDA kernel: {e}")
        # TODO: add metadata for compilation error? (show the error message)
        # clean up before returning
        del Model
        del ModelNew
        torch.cuda.empty_cache()
        return KernelExecResult(compiled=False, metadata=str(e)) # skip further steps

    with torch.no_grad():    
        set_seed(seed_num) # set seed for reproducible weights
        custom_model = ModelNew(*init_inputs)

        assert hasattr(custom_model, 'forward')  
        if verbose:
            print("[Eval] New Model with Custom CUDA Kernel Loaded")

    kernel_exec_result = None
    if measure_performance:
        if verbose:
            print("[Eval] Checking Both Correctness and Performance")
        raise NotImplementedError("Not implemented")
    else:
        if verbose:
            print("[Eval] Checking Correctness Only")
        try:
            is_correct = run_and_check_correctness(original_model, custom_model, get_inputs, num_times=num_times, verbose=verbose, seed=seed_num)
            kernel_exec_result = KernelExecResult(compiled=True, correctness=is_correct)
        except Exception as e:
            # TODO: add metadata for correctness error? e.g. runtime error, error in launching kernel, ...
            kernel_exec_result = KernelExecResult(compiled=True, correctness=False, metadata=str(e))
            # clean up before returning
            del Model
            del ModelNew
            torch.cuda.empty_cache()
            return kernel_exec_result
    # Clean up
    # delete ran-specific function definitions before next eval run
    del context
    # # release GPU memory
    torch.cuda.empty_cache()
    return kernel_exec_result
    

def run_and_check_correctness(original_model_instance: nn.Module, 
                              new_model_instance: nn.Module, 
                              get_inputs_fn: callable, 
                              num_times: int,
                              verbose=False, 
                              seed=42) -> tuple[bool, bool, float, float, float, float]:
    """
    run the model and check correctness, 
    assume model already loaded and compiled (loaded and compiled in the caller)
    this is all on GPU, requiring cuda device and transfer .cuda()
    """
    pass_count = 0

    with torch.no_grad():
        
        for trial in range(num_times):
            if verbose:
                print(f"[Eval] Generating Random Input with seed {seed}")
            
            set_seed(seed)
            inputs = get_inputs_fn()
            inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

            set_seed(seed)
            model = original_model_instance.cuda()

            set_seed(seed)
            model_new = new_model_instance.cuda()

            output = model(*inputs)
            torch.cuda.synchronize()

            try:
                output_new = model_new(*inputs)
                torch.cuda.synchronize()
                assert(output.shape == output_new.shape)
        
                is_correct = torch.allclose(output, output_new, atol=1e-03)
                pass_count += 1

            except Exception as e:
                # Count this as compilation issue
                # NOTE: something to discusss
                # Error in launching kernel for ModelNew CUDA error: invalid configuration argument
                # Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
                # NOTE: count runtime CUDA kernel as compile issue for now
                print(f"Error in launching kernel for ModelNew {e}")
                is_correct = False
                continue

        if verbose:
            if is_correct:
                print("[PASS] New Model matches Model")
            else:
                print("[FAIL] New Model does NOT match Model")
                # print("output from Model: ", output)
                # print("output from Model New: ", output_new)

    return pass_count == num_times

# if __name__ == "__main__":
#     fetch_kernel_from_database("kernelbench_prompt_v2_level_2", 1, 1, "http://localhost:9091")




# 