"""
This script tests the correctness of models in KernelBench by generating random inputs 
and random initialization. It compares the output of the original model against itself.
It ensures that the test is well-formed and there are no sources of non-determinism in the test.

Usage: 
    python verify_bench.py                              # Run all levels
    python verify_bench.py level=1                      # Run only level 1
    python verify_bench.py problem_ids=[96,100]         # Run only problem IDs 96 and 100
"""

import importlib.util
import os
import random

import numpy as np
import pydra
from pydra import Config
import torch

"""
Test all the reference architectures compiles 
and reproduce the same results when run against itself
"""

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def check_correctness(
    Model, NewModel, get_inputs, get_init_inputs, seed=42, atol=None, rtol=None, precision=None
):
    if atol is None:
        atol = get_tolerance_for_precision(precision)
    if rtol is None:
        rtol = get_tolerance_for_precision(precision)
    # run the model and check correctness
    with torch.no_grad():
        set_seed(seed)
        inputs = get_inputs()
        inputs = [x.cuda().to(precision) if isinstance(x, torch.Tensor) else x for x in inputs]

        for i, x in enumerate(inputs):
            if isinstance(x, torch.Tensor) and torch.isinf(x).any():
                raise ValueError(f"Input {i} contains infinity values")

        set_seed(seed)
        init_inputs = get_init_inputs()
        init_inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]

        set_seed(seed)
        model = Model(*init_inputs).cuda().to(precision)

        set_seed(seed)
        model_new = NewModel(*init_inputs).cuda().to(precision)

        output = model(*inputs)
        output_new = model_new(*inputs)

        if output.shape != output_new.shape:
            return False
        if not torch.allclose(output, output_new, atol=atol, rtol=rtol):
            return False
    return True


def run(Model, NewModel, get_inputs, get_init_inputs, seed=1012, precision=None):
    return check_correctness(Model, NewModel, get_inputs, get_init_inputs, seed, precision=precision)


from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import get_torch_dtype_from_string, get_tolerance_for_precision


class ScriptConfig(Config):
    def __init__(self):
        # Level(s) to run - can be single int or list
        self.level = [1, 2, 3]
        # Filter by problem IDs (e.g., [96, 100])
        self.problem_ids = []
        # Dataset source
        self.source = "local"
        # Precision: "fp32", "fp16", "bf16"
        self.precision = "fp32"


def run_all(level: int, problem_ids: list, source: str, precision: torch.dtype):
    """
    Run all problems in the given level.
    """
    
    print(f"Running Level {level} of length {len(problem_ids)} problems from {source} with precision {precision}")
    
    # Use problem_ids filtering at dataset level if specified
    if problem_ids:
        dataset = construct_kernelbench_dataset(level, source=source, problem_ids=problem_ids)
    else:
        dataset = construct_kernelbench_dataset(level, source=source)
    
    total = 0
    passed = 0
    fail_tests = []
    
    for problem in dataset:
        module_name = problem.name.replace(".py", "")
        total += 1
        try:
            problem_path = getattr(problem, "path", None)
            if not problem_path:
                raise ValueError(
                    f"Problem '{module_name}' does not have a local file path; "
                    "verify_bench.py only supports local datasets."
                )
            # Dynamically import the module
            spec = importlib.util.spec_from_file_location(
                module_name, problem_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Get the required attributes from the module
            Model = getattr(module, "Model")
            get_inputs = getattr(module, "get_inputs")
            get_init_inputs = getattr(module, "get_init_inputs")
            assert run(Model, Model, get_inputs, get_init_inputs, precision=precision)
            passed += 1
            print(f"Passed {module_name}")
        except Exception as e:
            print(f"Failed {module_name}: {e}")
            fail_tests.append(module_name)
    print(f"Level {level}: {passed}/{total} passed")
    if len(fail_tests) > 0:
        print(f"Failed tests: {fail_tests}")


@pydra.main(base=ScriptConfig)
def main(config: ScriptConfig):
    levels = config.level if isinstance(config.level, list) else [config.level]
    problem_ids = config.problem_ids if config.problem_ids else []
    precision = get_torch_dtype_from_string(config.precision)
    
    for level in levels:
        run_all(level, problem_ids, config.source, precision)


if __name__ == "__main__":
    main()
