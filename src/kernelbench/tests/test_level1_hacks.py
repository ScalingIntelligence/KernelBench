import os
import torch
import pydra
from pydra import Config

from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
from kernelbench.utils import read_file

TESTS_DIR = os.path.dirname(__file__)
PROBLEMS_DIR = os.path.join(TESTS_DIR, "problems")
SOLUTIONS_DIR = os.path.join(TESTS_DIR, "solutions")

PROBLEMS = ["94_MSELoss", "96_HuberLoss", "100_HingeLoss"]


class ScriptConfig(Config):
    def __init__(self):
        self.precision = "fp32"


def evaluate(problem_src: str, solution_src: str, precision: torch.dtype) -> bool:
    if not solution_src.strip():
        return None
    result = eval_kernel_against_ref(
        original_model_src=problem_src,
        custom_model_src=solution_src,
        measure_performance=False,
        verbose=False,
        precision=precision,
        num_correct_trials=5,
    )
    return result.correctness if result else False


@pydra.main(base=ScriptConfig)
def main(config: ScriptConfig):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    precision = get_torch_dtype_from_string(config.precision)

    for problem in PROBLEMS:
        old_src = read_file(os.path.join(PROBLEMS_DIR, f"{problem}_OLD.py"))
        uniform_normal_src = read_file(os.path.join(PROBLEMS_DIR, f"{problem}_uniform_normal.py"))
        pareto_src = read_file(os.path.join(PROBLEMS_DIR, f"{problem}_pareto.py"))

        analytical_hack_src = read_file(os.path.join(SOLUTIONS_DIR, f"{problem}_analytical_hack.py"))
        partial_hack_src = read_file(os.path.join(SOLUTIONS_DIR, f"{problem}_partial_computation_hack.py"))
        correct_src = read_file(os.path.join(SOLUTIONS_DIR, f"{problem}_correct.py"))

        print(f"\n{'='*60}")
        print(f"Testing {problem}")
        print(f"{'='*60}")

        def check(problem_name, problem_src):
            correct_result = evaluate(problem_src, correct_src, precision)
            analytical_result = evaluate(problem_src, analytical_hack_src, precision)
            partial_result = evaluate(problem_src, partial_hack_src, precision)

            print(f"  {problem_name}:")
            for name, result in [
                ("CORRECT", correct_result),
                ("ANALYTICAL_HACK", analytical_result),
                ("PARTIAL_HACK", partial_result),
            ]:
                if result is None:
                    print(f"      SKIPPED: {name} (empty solution file)")
                    continue
                action = "accepts" if result else "rejects"
                print(f"      {action} {name}")

        check("OLD", old_src)
        check("uniform_normal", uniform_normal_src)
        check("pareto", pareto_src)


if __name__ == "__main__":
    main()

