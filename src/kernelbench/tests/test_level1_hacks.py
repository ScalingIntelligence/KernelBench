import os
import torch

from kernelbench.eval import eval_kernel_against_ref
from kernelbench.utils import read_file

TESTS_DIR = os.path.dirname(__file__)
PROBLEMS_DIR = os.path.join(TESTS_DIR, "problems")
SOLUTIONS_DIR = os.path.join(TESTS_DIR, "solutions")

PROBLEMS = ["94_MSELoss", "96_HuberLoss", "100_HingeLoss"]


def evaluate(problem_src: str, solution_src: str) -> bool:
    if not solution_src.strip():
        return None
    result = eval_kernel_against_ref(
        original_model_src=problem_src,
        custom_model_src=solution_src,
        measure_performance=False,
        verbose=False,
    )
    return result.correctness if result else False


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    all_passed = True

    for problem in PROBLEMS:
        old_src = read_file(os.path.join(PROBLEMS_DIR, f"{problem}_OLD.py"))
        new_src = read_file(os.path.join(PROBLEMS_DIR, f"{problem}_NEW.py"))
        hack_src = read_file(os.path.join(SOLUTIONS_DIR, f"{problem}_HACK.py"))
        correct_src = read_file(os.path.join(SOLUTIONS_DIR, f"{problem}_CORRECT.py"))

        print(f"\n{'='*60}")
        print(f"Testing {problem}")
        print(f"{'='*60}")

        hack_vs_old = evaluate(old_src, hack_src)
        correct_vs_old = evaluate(old_src, correct_src)
        hack_vs_new = evaluate(new_src, hack_src)
        correct_vs_new = evaluate(new_src, correct_src)

        def check(solution_name, problem_name, correctness, should_be_correct):
            nonlocal all_passed
            if correctness is None:
                print(f"  SKIPPED: {solution_name} (empty solution file)")
                return
            correct_str = "correct" if correctness else "incorrect"
            expected_str = "correct" if should_be_correct else "incorrect"
            test_passed = correctness == should_be_correct
            status = "✓" if test_passed else "✗"
            if not test_passed:
                all_passed = False
            print(f"  {status} {solution_name} on {problem_name}: {correct_str} (expected {expected_str})")

        check("HACK", "OLD problem", hack_vs_old, True)
        check("CORRECT", "OLD problem", correct_vs_old, True)
        check("HACK", "NEW problem", hack_vs_new, False)
        check("CORRECT", "NEW problem", correct_vs_new, True)

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

