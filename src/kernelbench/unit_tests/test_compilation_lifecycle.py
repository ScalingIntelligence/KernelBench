"""Exercise failed compilation through batch evaluation and result persistence."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REFERENCE = """
import torch
from torch import nn

class Model(nn.Module):
    def forward(self, a, b):
        return a + b

def get_inputs():
    return [torch.randn(16), torch.randn(16)]

def get_init_inputs():
    return []
"""


def run_batch(directory, backend):
    import multiprocessing

    repository = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository / "scripts"))
    from eval_from_generations import EvalConfig, batch_eval

    from kernelbench.dataset import LocalKernelBenchDataset

    multiprocessing.set_start_method("spawn")
    directory.mkdir(parents=True, exist_ok=True)
    level = directory / "dataset" / "level1"
    level.mkdir(parents=True)
    generated = directory / "generated"
    generated.mkdir()
    kernels = [
        "tile_size\n",
        "block_size\n",
        f"open({str(directory / 'missing-header.h')!r})\n",
        "def invalid syntax\n",
        REFERENCE.replace("class Model(", "class ModelNew("),
    ]
    for problem_id, source in enumerate(kernels, start=1):
        (level / f"{problem_id}_addition.py").write_text(REFERENCE)
        (generated / f"level_1_problem_{problem_id}_sample_0_kernel.py").write_text(
            source
        )
    dataset = LocalKernelBenchDataset(level=1, base_path=str(level.parent))
    config = EvalConfig()
    config.run_name = "compilation-lifecycle"
    config.dataset_src = "local"
    config.level = 1
    config.backend = backend
    config.num_gpu_devices = 1
    config.measure_performance = False
    config.num_correct_trials = 2
    config.timeout = 90
    config.kernel_eval_build_dir = str(directory / "build")
    batch_eval(
        [(problem_id, 0) for problem_id in range(1, len(kernels) + 1)],
        config,
        dataset,
        str(generated),
        str(directory / "eval_results.json"),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA evaluation requires a GPU"
)
@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_compilation_failures_leave_the_next_sample_runnable(tmp_path, backend):
    temporary_modules = tmp_path / "modules"
    temporary_modules.mkdir()
    completed = subprocess.run(
        [sys.executable, __file__, "--worker", str(tmp_path / "run"), backend],
        check=False,
        env={**os.environ, "TMPDIR": str(temporary_modules)},
        capture_output=True,
        text=True,
        timeout=240,
    )
    if completed.returncode:
        pytest.fail(completed.stdout + completed.stderr)
    results = json.loads((tmp_path / "run" / "eval_results.json").read_text())
    if sorted(results) != ["1", "2", "3", "4", "5"]:
        pytest.fail("A compilation failure prevented later samples from being recorded")
    for problem_id in range(1, 5):
        record = results[str(problem_id)][0]
        if record["compiled"] or record["correctness"] or record["runtime"] != -1:
            pytest.fail(f"Failed sample {problem_id} was recorded as runnable")
        if not record["metadata"].get("compilation_error"):
            pytest.fail(f"Failed sample {problem_id} lost its compilation diagnostic")
    if "block_size" not in results["2"][0]["metadata"]["compilation_error"]:
        pytest.fail("The undefined variable diagnostic was lost during persistence")
    valid = results["5"][0]
    if not valid["compiled"] or not valid["correctness"]:
        pytest.fail("The valid sample following compilation failures did not execute")
    if list(temporary_modules.glob("*.py")) or list(temporary_modules.rglob("*.pyc")):
        pytest.fail("Module evaluation left temporary source or bytecode files behind")


if __name__ == "__main__" and sys.argv[1] == "--worker":
    run_batch(Path(sys.argv[2]), sys.argv[3])
