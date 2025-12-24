import argparse
import os
import sys

# Add repo root to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
from scripts.shinka_evolve.make_seed import create_seed_file
from src.prompt_constructor_toml import get_prompt_for_backend

def build_kernelbench_sys_msg(model_name):
    # Get standard KernelBench prompt context
    base_prompt = get_prompt_for_backend(
        ref_arch_src="", 
        backend="cuda",
        option="few_shot", 
        precision="fp32",
        include_hardware=True,
        gpu_name="H100"
    )
    
    # Remove empty placeholder
    base_prompt = base_prompt.replace("You are given the following architecture:\n\n\n\n\n", "")
    
    sys_msg = f"You are a world-class CUDA optimization expert.\n\n{base_prompt}"
    
    # ADD CRITICAL ORDERING INSTRUCTIONS
    sys_msg += """
    
    # CRITICAL INSTRUCTIONS FOR SHINKA EVOLUTION
    
    1. **ORDERING IS VITAL:** Python executes top-to-bottom.
       - You MUST define your `cuda_source` string FIRST.
       - You MUST define `cpp_source` SECOND.
       - You MUST call `load_inline` THIRD (assigning it to a variable like `my_kernel`).
       - You MUST define `class ModelNew` LAST.
       
    2. **PLACEMENT:** 
       - Look for the comment `# --- INSERT CUSTOM CUDA KERNEL HERE ---`.
       - Place your C++ string definitions and `load_inline` call there.
       
    3. **CLASS USAGE:**
       - Inside `ModelNew.__init__`, assign the global kernel object to `self` (e.g., `self.kernel = my_kernel`).
       - Do NOT define `load_inline` inside the class methods.
       
    4. **SYNTAX:**
       - Use `super().__init__()` in the constructor.
    """
    
    return sys_msg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--problem_id", type=int, default=1)
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06") 
    parser.add_argument("--generations", type=int, default=10)
    args = parser.parse_args()

    # 1. Setup Workspace
    run_name = f"shinka_L{args.level}_P{args.problem_id}_{args.model.replace('/', '_')}"
    results_dir = f"runs/{run_name}"
    os.makedirs(results_dir, exist_ok=True)

    # 2. Generate Initial Program
    init_path = os.path.join(results_dir, "initial.py")
    create_seed_file(args.level, args.problem_id, init_path)

    # 3. Configure Shinka
    job_config = LocalJobConfig(
        eval_program_path="scripts/shinka_evolve/evaluate_bridge.py",
        extra_cmd_args={
            "level": args.level,
            "problem_id": args.problem_id
        }
    )

    db_config = DatabaseConfig(
        db_path="evolution.db",
        parent_selection_strategy="weighted", 
        archive_size=20
    )

    evo_config = EvolutionConfig(
        task_sys_msg=build_kernelbench_sys_msg(args.model),
        num_generations=args.generations,
        max_parallel_jobs=1,
        init_program_path=init_path,
        results_dir=results_dir,
        llm_models=[args.model],
        use_text_feedback=True,
        # We increase Full Rewrite probability because ordering is hard to fix with Diff
        patch_types=["diff", "full"], 
        patch_type_probs=[0.4, 0.6], 
        language="python" 
    )

    runner = EvolutionRunner(evo_config, job_config, db_config)
    runner.run()

if __name__ == "__main__":
    main()