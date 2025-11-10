"""
Run Kernel Search - Main entry point for CUDA kernel optimization with AIDE tree search
Integrates KernelBench evaluation with AIDE's tree search framework
"""

import atexit
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.status import Status
from rich.text import Text
from rich.tree import Tree

from datasets import load_dataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from journal import Journal
from journal2report import journal2report
from kernel_agent import KernelAgent
from kernel_interpreter import KernelInterpreter
from utils.config import save_run
from utils.tree_export import generate as generate_tree_viz
from utils.serialize import dump_json

from src.dataset import construct_kernelbench_dataset
from src.utils import read_file, set_gpu_arch

logger = logging.getLogger("aide")


def journal_to_rich_tree(journal: Journal):
    """Generate a rich tree visualization of the solution tree."""
    # Handle empty journal
    if len(journal) == 0:
        tree = Tree("[bold blue]Kernel Optimization Tree")
        tree.add("[dim]No nodes yet[/dim]")
        return tree
    
    best_node = journal.get_best_node()

    def append_rec(node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""
            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} ms (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f} ms"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Kernel Optimization Tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def load_kernel_config(config_path: str = None):
    """Load kernel search configuration."""
    # Check if config path is specified via environment variable
    if config_path is None:
        config_path = os.environ.get('KERNEL_SEARCH_CONFIG')
    
    if config_path is None:
        config_path = Path(__file__).parent / "kernel_config.yaml"
    
    cfg = OmegaConf.load(config_path)
    
    # Merge with CLI arguments
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_conf)
    
    return cfg


def prep_kernel_config(cfg):
    """Prepare and validate kernel search configuration."""
    # Set up logging directories
    log_dir = Path(cfg.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    workspace_dir = Path(cfg.workspace_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment name with index
    def get_next_logindex(dir_path: Path) -> int:
        max_index = -1
        if not dir_path.exists():
            return 0
        for p in dir_path.iterdir():
            try:
                current_index = int(p.name.split("-")[0])
                if current_index > max_index:
                    max_index = current_index
            except (ValueError, IndexError):
                pass
        return max_index + 1
    
    ind = max(get_next_logindex(log_dir), get_next_logindex(workspace_dir))
    
    # Generate experiment name
    if cfg.exp_name is None:
        import coolname
        cfg.exp_name = coolname.generate_slug(3)
    
    cfg.exp_name = f"{ind}-{cfg.exp_name}"
    
    cfg.log_dir = (log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (workspace_dir / cfg.exp_name).resolve()
    
    # Create directories
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.workspace_dir.mkdir(parents=True, exist_ok=True)
    
    return cfg


def load_reference_architecture(cfg):
    """Load reference PyTorch architecture from KernelBench dataset."""
    if cfg.kernel.dataset_src == "huggingface":
        dataset = load_dataset(cfg.kernel.get("dataset_name", "ScalingIntelligence/KernelBench"))
        curr_level_dataset = dataset[f"level_{cfg.kernel.level}"]
        
        curr_problem_row = curr_level_dataset.filter(
            lambda x: x["problem_id"] == cfg.kernel.problem_id
        )
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]
        
    elif cfg.kernel.dataset_src == "local":
        # Construct dataset from local files
        curr_level_dataset = construct_kernelbench_dataset(cfg.kernel.level)
        
        problem_idx = cfg.kernel.problem_id - 1  # 0-indexed
        ref_arch_path = curr_level_dataset[problem_idx]
        
        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)
    else:
        raise ValueError(f"Unknown dataset_src: {cfg.kernel.dataset_src}")
    
    # Validate problem number matches
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == cfg.kernel.problem_id, \
        f"Problem number mismatch: {problem_number} != {cfg.kernel.problem_id}"
    
    return ref_arch_src, problem_name


def generate_task_description(cfg, problem_name: str) -> dict:
    """Generate task description from problem metadata."""
    task_desc = {
        "Problem": problem_name.replace("_", " ").replace(".py", ""),
        "Level": cfg.kernel.level,
        "Backend": cfg.gpu.backend,
        "Precision": cfg.gpu.precision,
        "Goal": (
            "Optimize the given PyTorch architecture by implementing custom CUDA kernels. "
            "Your implementation should be correct (match reference outputs) and fast (minimize runtime)."
        ),
        "Evaluation": (
            f"Kernels are evaluated through: "
            f"(1) Compilation check, "
            f"(2) Correctness check with {cfg.evaluation.num_correct_trials} random input trials, "
            f"(3) Performance measurement over {cfg.evaluation.num_perf_trials} trials. "
            f"Success metric: runtime in milliseconds (lower is better)."
        ),
    }
    return task_desc


def format_task_desc_as_markdown(task_desc: dict) -> str:
    """Format task description as markdown string."""
    lines = []
    for key, value in task_desc.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"**{key}:**")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"**{key}:** {value}")
    return "\n".join(lines)


def run_kernel_search():
    """Main entry point for kernel optimization with tree search."""
    # Load and prepare configuration
    cfg = load_kernel_config()
    cfg = prep_kernel_config(cfg)
    
    logger.info(f'Starting kernel search "{cfg.exp_name}"')
    print(f"\n{'='*80}")
    print(f"🚀 CUDA Kernel Optimization with AIDE Tree Search")
    print(f"{'='*80}")
    print(f"Experiment: {cfg.exp_name}")
    print(f"Level {cfg.kernel.level}, Problem {cfg.kernel.problem_id}")
    print(f"Backend: {cfg.gpu.backend}, Precision: {cfg.gpu.precision}")
    print(f"Search steps: {cfg.agent.steps}")
    print(f"{'='*80}\n")
    
    # Set GPU architecture for compilation
    if cfg.gpu.arch:
        set_gpu_arch(cfg.gpu.arch)
    
    # Load reference architecture
    with Status("Loading reference architecture..."):
        ref_arch_src, problem_name = load_reference_architecture(cfg)
    
    print(f"✓ Loaded problem: {problem_name}")
    
    # Generate task description
    task_desc = generate_task_description(cfg, problem_name)
    task_desc_str = format_task_desc_as_markdown(task_desc)
    
    # Save reference architecture
    with open(cfg.log_dir / "reference_architecture.py", "w") as f:
        f.write(ref_arch_src)
    
    # Initialize journal for solution tree
    journal = Journal()
    
    # Initialize kernel interpreter
    interpreter = KernelInterpreter(
        ref_arch_src=ref_arch_src,
        working_dir=cfg.workspace_dir,
        device=torch.cuda.current_device() if torch.cuda.is_available() else None,
        backend=cfg.gpu.backend,
        precision=cfg.gpu.precision,
        num_correct_trials=cfg.evaluation.num_correct_trials,
        num_perf_trials=cfg.evaluation.num_perf_trials,
        measure_performance=cfg.evaluation.measure_performance,
        timeout=cfg.evaluation.timeout,
        verbose=False,
    )
    
    # Initialize kernel agent
    debug_mode = cfg.get('debug', False)
    if debug_mode:
        print(f"\n[DEBUG] Initializing KernelAgent...")
    agent = KernelAgent(
        ref_arch_src=ref_arch_src,
        task_desc=task_desc_str,
        cfg=cfg,
        journal=journal,
        backend=cfg.gpu.backend,
        precision=cfg.gpu.precision,
    )
    if debug_mode:
        print(f"[DEBUG] KernelAgent initialized")
        print(f"[DEBUG] Agent type: {type(agent)}")
        print(f"[DEBUG] Agent has step method: {hasattr(agent, 'step')}")
        print(f"[DEBUG] Agent journal: {agent.journal}")
        print(f"[DEBUG] Journal length: {len(journal)}")
    
    # Cleanup on exit
    def cleanup():
        if len(journal) == 0:
            shutil.rmtree(cfg.workspace_dir, ignore_errors=True)
    
    atexit.register(cleanup)
    
    # Execution callback for agent
    def exec_callback(code: str, reset_session: bool = True):
        return interpreter.run(code, reset_session)
    
    # Progress tracking
    global_step = len(journal)
    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    status = Status("[green]Generating kernel code...")
    prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)
    
    def generate_live():
        """Generate live display for progress tracking."""
        tree = journal_to_rich_tree(journal)
        prog.update(prog.task_ids[0], completed=global_step)
        
        info_lines = [
            f"Problem: [yellow]{problem_name}[/yellow]",
            f"Backend: [yellow]{cfg.gpu.backend}[/yellow], Precision: [yellow]{cfg.gpu.precision}[/yellow]",
            f"Best kernel: [yellow]{journal.get_best_node().metric.value:.3f} ms[/yellow]" if journal.good_nodes else "No successful kernels yet",
            "",
            f"Log: [yellow]▶ {str(cfg.log_dir)}[/yellow]",
        ]
        
        # Create a condensed task description for display
        task_lines = [
            f"[bold]Problem:[/bold] {problem_name.replace('_', ' ').replace('.py', '')}",
            f"[bold]Level:[/bold] {cfg.kernel.level} | [bold]Backend:[/bold] {cfg.gpu.backend} | [bold]Precision:[/bold] {cfg.gpu.precision}",
        ]
        task_desc_condensed = "\n".join(task_lines)
        
        left = Group(
            Panel(Text(task_desc_condensed.strip()), title="Task", height=6),
            prog,
            status,
        )
        right = tree
        info = Group(*[Text(line) for line in info_lines])
        
        return Panel(
            Group(
                Padding(info, (1, 1, 1, 1)),
                Columns(
                    [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                    equal=False,
                    expand=True,
                ),
            ),
            title=f'[b]AIDE Kernel Search: [bold green]"{cfg.exp_name}[/b]"',
            subtitle="Press [b]Ctrl+C[/b] to stop",
        )
    
    # Main search loop
    debug_mode = cfg.get('debug', False)
    
    if debug_mode:
        print(f"\n[DEBUG] Starting search loop:")
        print(f"  - global_step: {global_step}")
        print(f"  - cfg.agent.steps: {cfg.agent.steps}")
        print(f"  - Loop condition: {global_step} < {cfg.agent.steps} = {global_step < cfg.agent.steps}")
    
    # Choose execution mode based on debug flag
    if not debug_mode:
        # Clean Live UI mode (default)
        with Live(
            generate_live(),
            refresh_per_second=4,
            screen=True,  # Full-screen mode for clean UI
        ) as live:
            iteration = 0
            while global_step < cfg.agent.steps:
                iteration += 1
                try:
                    agent.step(exec_callback=exec_callback)
                    
                    # Save progress
                    save_run(cfg, journal)
                    global_step = len(journal)
                    
                    # Update live display
                    live.update(generate_live())
                    
                except KeyboardInterrupt:
                    print("\n\n⚠️  Search interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error during search step: {e}", exc_info=True)
                    print(f"\n\n❌ Error during search: {e}")
                    break
    else:
        # Debug mode with verbose output
        iteration = 0
        while global_step < cfg.agent.steps:
            iteration += 1
            print(f"\n[DEBUG] Loop iteration {iteration}, global_step={global_step}")
            try:
                print(f"[DEBUG] Calling agent.step()...")
                agent.step(exec_callback=exec_callback)
                print(f"[DEBUG] agent.step() completed")
                
                # Save progress
                print(f"[DEBUG] Saving progress...")
                save_run(cfg, journal)
                
                global_step = len(journal)
                print(f"[DEBUG] Updated global_step to {global_step}, journal length: {len(journal)}")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Search interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error during search step: {e}", exc_info=True)
                print(f"\n\n❌ Error during search: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"\n[DEBUG] Exited search loop after {iteration} iterations")
    
    interpreter.cleanup_session()
    
    # Final save
    save_run(cfg, journal)
    
    # Print results summary
    print(f"\n{'='*80}")
    print(f"🏁 Kernel Search Complete")
    print(f"{'='*80}")
    print(f"Total nodes explored: {len(journal)}")
    print(f"Successful kernels: {len(journal.good_nodes)}")
    print(f"Failed kernels: {len(journal.buggy_nodes)}")
    
    if journal.good_nodes:
        best_node = journal.get_best_node()
        print(f"\n🏆 Best kernel performance: {best_node.metric.value:.3f} ms")
        print(f"   Plan: {best_node.plan[:100]}...")
        
        # Save best kernel
        with open(cfg.log_dir / "best_kernel.py", "w") as f:
            f.write(best_node.code)
        print(f"\n✓ Best kernel saved to: {cfg.log_dir / 'best_kernel.py'}")
    else:
        print(f"\n⚠️  No successful kernels found")
    
    print(f"\n📊 Visualization: {cfg.log_dir / 'tree_plot.html'}")
    print(f"📁 Full results: {cfg.log_dir}")
    print(f"{'='*80}\n")
    
    # Generate report if configured
    if cfg.generate_report and journal.good_nodes:
        print("Generating optimization report...")
        try:
            report = journal2report(journal, task_desc, cfg.report)
            report_path = cfg.log_dir / "optimization_report.md"
            with open(report_path, "w") as f:
                f.write(report)
            print(f"✓ Report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            print(f"⚠️  Failed to generate report: {e}")


if __name__ == "__main__":
    run_kernel_search()
