#!/usr/bin/env python3
"""
Helper script to run kernel search with simplified command-line interface
Example usage:
    python run_kernel_search_simple.py --level 1 --problem_id 1 --steps 10
    python run_kernel_search_simple.py --level 1 --problem_id 19 --backend cuda --precision fp32 --steps 20
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from the package
sys.path.insert(0, str(Path(__file__).parent))

from run_kernel_search import run_kernel_search
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description="Run CUDA kernel optimization with AIDE tree search")
    
    # Problem specification
    parser.add_argument("--level", type=int, default=1, help="KernelBench level (1-4)")
    parser.add_argument("--problem_id", type=int, default=1, help="Problem ID within level")
    parser.add_argument("--dataset_src", type=str, default="local", choices=["local", "huggingface"],
                       help="Dataset source")
    
    # GPU settings
    parser.add_argument("--backend", type=str, default="cuda", 
                       choices=["cuda", "triton", "tilelang", "cute"],
                       help="Backend for kernel implementation")
    parser.add_argument("--precision", type=str, default="fp32",
                       choices=["fp32", "fp16", "bf16"],
                       help="Computation precision")
    parser.add_argument("--gpu_arch", type=str, nargs="+", default=["Ada"],
                       help="GPU architecture for compilation (e.g., Ada, Ampere, Volta)")
    
    # Search configuration
    parser.add_argument("--steps", type=int, default=20, help="Number of search steps")
    parser.add_argument("--num_drafts", type=int, default=3, help="Number of initial drafts")
    parser.add_argument("--debug_prob", type=float, default=0.3, help="Probability of debugging")
    parser.add_argument("--max_debug_depth", type=int, default=2, help="Max consecutive debug attempts")
    
    # Evaluation settings
    parser.add_argument("--num_correct_trials", type=int, default=5,
                       help="Number of correctness trials")
    parser.add_argument("--num_perf_trials", type=int, default=100,
                       help="Number of performance trials")
    parser.add_argument("--no_perf", action="store_true",
                       help="Skip performance measurement")
    
    # LLM settings
    parser.add_argument("--server_type", type=str, default="google",
                       help="LLM server type (google, openai, anthropic, etc.)")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name (uses preset if not specified)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=8192,
                       help="Max tokens to generate")
    
    # Reasoning model settings
    parser.add_argument("--is_reasoning_model", action="store_true",
                       help="Use reasoning model mode (o1, o3, etc.)")
    parser.add_argument("--reasoning_effort", type=str, default=None,
                       choices=["low", "medium", "high"],
                       help="Reasoning effort for o1/o3 models")
    parser.add_argument("--budget_tokens", type=int, default=0,
                       help="Budget tokens for Claude extended thinking")
    
    # Output settings
    parser.add_argument("--exp_name", type=str, default=None,
                       help="Experiment name (auto-generated if not specified)")
    parser.add_argument("--log_dir", type=str, default="./logs/kernel_search",
                       help="Log directory")
    parser.add_argument("--workspace_dir", type=str, default="./workspaces/kernel_search",
                       help="Workspace directory")
    parser.add_argument("--no_report", action="store_true",
                       help="Skip report generation")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output (disables live UI)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build OmegaConf configuration from arguments
    cfg = OmegaConf.create({
        "kernel": {
            "dataset_src": args.dataset_src,
            "level": args.level,
            "problem_id": args.problem_id,
        },
        "gpu": {
            "arch": args.gpu_arch,
            "precision": args.precision,
            "backend": args.backend,
        },
        "evaluation": {
            "num_correct_trials": args.num_correct_trials,
            "num_perf_trials": args.num_perf_trials,
            "measure_performance": not args.no_perf,
            "timeout": 600,
        },
        "inference": {
            "server_type": args.server_type,
            "model_name": args.model_name,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "is_reasoning_model": args.is_reasoning_model,
            "reasoning_effort": args.reasoning_effort,
            "budget_tokens": args.budget_tokens,
        },
        "agent": {
            "steps": args.steps,
            "code": {
                "model": args.model_name if args.model_name else "gemini/gemini-2.5-flash",
                "temp": args.temperature,
            },
            "feedback": {
                "model": args.model_name if args.model_name else "gemini/gemini-2.5-flash",
                "temp": 0.3,
            },
            "search": {
                "num_drafts": args.num_drafts,
                "debug_prob": args.debug_prob,
                "max_debug_depth": args.max_debug_depth,
            },
            "k_fold_validation": 1,
            "expose_prediction": False,
            "data_preview": False,
        },
        "exec": {
            "timeout": 600,
            "agent_file_name": "kernel_code.py",
            "format_tb_ipython": False,
        },
        "exp_name": args.exp_name,
        "log_dir": args.log_dir,
        "workspace_dir": args.workspace_dir,
        "generate_report": not args.no_report,
        "debug": args.debug,
        "report": {
            "model": args.model_name if args.model_name else "gemini/gemini-2.5-flash",
            "temp": 0.5,
        },
        "data_dir": None,
        "desc_file": None,
        "goal": None,
        "eval": None,
        "preprocess_data": False,
        "copy_data": False,
    })
    
    # Temporarily override sys.argv to pass config via OmegaConf
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]  # Keep only script name
    
    # Save config to a temporary location and load it
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        OmegaConf.save(cfg, f)
        temp_config_path = f.name
    
    try:
        # Set environment variable to use our temp config
        os.environ['KERNEL_SEARCH_CONFIG'] = temp_config_path
        run_kernel_search()
    finally:
        # Restore sys.argv
        sys.argv = original_argv
        # Clean up temp file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == "__main__":
    main()
