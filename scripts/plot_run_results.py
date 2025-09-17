import os
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

# Lazy import matplotlib inside functions to avoid hard dependency at import time.


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return None


def _find_latest_run(base_runs_dir: str, level: int, problem_id: int) -> Optional[str]:
    prefix = f"iterative_l{level}_p{problem_id}_"
    candidates = [d for d in os.listdir(base_runs_dir) if d.startswith(prefix)]
    if not candidates:
        return None
    candidates.sort()  # timestamp suffix sort
    return os.path.join(base_runs_dir, candidates[-1])


def _collect_run_data(run_dir: str) -> Dict[str, Any]:
    level_dir = os.path.join(run_dir, "level_1")
    # Detect level folder dynamically
    for name in os.listdir(run_dir):
        if name.startswith("level_") and os.path.isdir(os.path.join(run_dir, name)):
            level_dir = os.path.join(run_dir, name)
            break

    problem_dir = None
    for name in os.listdir(level_dir):
        p = os.path.join(level_dir, name)
        if name.startswith("problem_") and os.path.isdir(p):
            problem_dir = p
            break
    assert problem_dir, f"No problem_* directory found under {level_dir}"

    summary = _read_json(os.path.join(problem_dir, "summary.json")) or {}
    baseline_eval = _read_json(os.path.join(problem_dir, "baseline_eval.json")) or {}

    # Samples
    samples_dir = os.path.join(problem_dir, "samples")
    samples: List[Dict[str, Any]] = []
    if os.path.isdir(samples_dir):
        for name in sorted(os.listdir(samples_dir)):
            sp = os.path.join(samples_dir, name)
            if not os.path.isdir(sp):
                continue
            ej = _read_json(os.path.join(sp, "eval.json")) or {}
            rt = (ej.get("runtime_stats") or {}).get("mean")
            temp = (ej.get("metadata") or {}).get("sample_temperature")
            samples.append({
                "sample": name,
                "temperature": temp,
                "runtime_mean": rt,
                "eval": ej,
            })

    # Rounds
    rounds_dir = os.path.join(problem_dir, "rounds")
    rounds: List[Dict[str, Any]] = []
    if os.path.isdir(rounds_dir):
        for name in sorted(os.listdir(rounds_dir)):
            rp = os.path.join(rounds_dir, name)
            if not os.path.isdir(rp):
                continue
            ej = _read_json(os.path.join(rp, "eval.json")) or {}
            rt = (ej.get("runtime_stats") or {}).get("mean")
            rounds.append({
                "round": name,
                "round_idx": _safe_int(name.split("_")[-1]),
                "runtime_mean": rt,
                "eval": ej,
            })

    return {
        "run_dir": run_dir,
        "problem_dir": problem_dir,
        "summary": summary,
        "baseline": baseline_eval,
        "samples": samples,
        "rounds": rounds,
    }


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_results(run_data: Dict[str, Any], out_dir: Optional[str] = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Please pip install matplotlib to generate plots.")
        return

    problem_dir = run_data["problem_dir"]
    out_dir = out_dir or os.path.join(problem_dir, "analysis")
    _ensure_dir(out_dir)

    baseline_mean = (run_data.get("baseline") or {}).get("runtime_stats", {}).get("mean")
    best_mean = (run_data.get("summary") or {}).get("stages", {}).get("best", {}).get("runtime_stats", {}).get("mean")
    oneshot_mean = (run_data.get("summary") or {}).get("stages", {}).get("oneshot", {}).get("runtime_stats", {}).get("mean")

    # 1) Temperature vs Runtime (Samples)
    temps = []
    rts = []
    labels = []
    for s in run_data["samples"]:
        if s["temperature"] is not None and s["runtime_mean"] is not None:
            temps.append(float(s["temperature"]))
            rts.append(float(s["runtime_mean"]))
            labels.append(s["sample"])

    if temps:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(temps, rts, marker="o", linestyle="-", label="samples")
        if baseline_mean is not None:
            ax.axhline(float(baseline_mean), color="red", linestyle="--", label="baseline")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Mean runtime (ms)")
        ax.set_title("Temperature sweep vs runtime")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "samples_temperature_vs_runtime.png"), dpi=150)
        plt.close(fig)

    # 2) Refinement rounds vs Runtime
    round_idx = []
    round_rt = []
    for r in run_data["rounds"]:
        if r["round_idx"] is not None and r["runtime_mean"] is not None:
            round_idx.append(int(r["round_idx"]))
            round_rt.append(float(r["runtime_mean"]))

    if round_idx:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(round_idx, round_rt, marker="o", linestyle="-", label="refinement rounds")
        if baseline_mean is not None:
            ax.axhline(float(baseline_mean), color="red", linestyle="--", label="baseline")
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean runtime (ms)")
        ax.set_title("Iterative refinement trajectory")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "refinement_rounds_vs_runtime.png"), dpi=150)
        plt.close(fig)

    # 3) Bars: baseline vs oneshot vs best
    bars = []
    heights = []
    if baseline_mean is not None:
        bars.append("baseline")
        heights.append(float(baseline_mean))
    if oneshot_mean is not None and ("baseline" not in bars or float(oneshot_mean) != heights[bars.index("baseline")]):
        bars.append("oneshot")
        heights.append(float(oneshot_mean))
    if best_mean is not None:
        bars.append("best")
        heights.append(float(best_mean))

    if bars:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(bars, heights, color=["#999", "#77c", "#4c4"][: len(bars)])
        ax.set_ylabel("Mean runtime (ms)")
        ax.set_title("Baseline vs Best")
        for i, v in enumerate(heights):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "baseline_vs_best.png"), dpi=150)
        plt.close(fig)

    # 4) Save CSV-like summaries for further analysis
    with open(os.path.join(out_dir, "samples_summary.csv"), "w") as f:
        f.write("sample,temperature,runtime_mean\n")
        for s in run_data["samples"]:
            f.write(f"{s['sample']},{s['temperature']},{s['runtime_mean']}\n")

    with open(os.path.join(out_dir, "rounds_summary.csv"), "w") as f:
        f.write("round,round_idx,runtime_mean\n")
        for r in run_data["rounds"]:
            f.write(f"{r['round']},{r['round_idx']},{r['runtime_mean']}\n")


def main():
    parser = argparse.ArgumentParser(description="Plot graphs from a KernelBench iterative run")
    parser.add_argument("--runs_dir", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs"), help="Root runs directory")
    parser.add_argument("--level", type=int, default=1, help="Level number")
    parser.add_argument("--problem_id", type=int, default=1, help="Problem ID")
    parser.add_argument("--run_path", default=None, help="Path to a specific run directory (overrides level/problem selection)")
    parser.add_argument("--out_dir", default=None, help="Custom output directory for plots (defaults to run/problem/analysis)")
    args = parser.parse_args()

    if args.run_path:
        run_dir = args.run_path
    else:
        run_dir = _find_latest_run(args.runs_dir, args.level, args.problem_id)
        if not run_dir:
            raise SystemExit(f"No runs found under {args.runs_dir} for level={args.level} problem_id={args.problem_id}")

    data = _collect_run_data(run_dir)
    plot_results(data, out_dir=args.out_dir)
    print(f"Saved plots and summaries under: {args.out_dir or os.path.join(data['problem_dir'], 'analysis')}")


if __name__ == "__main__":
    main()
