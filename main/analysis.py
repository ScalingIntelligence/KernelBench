from argparse import ArgumentParser
import os
import json
import matplotlib.pyplot as plt
import numpy as np

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(REPO_TOP_DIR, "plots")
RUNS_DIR = os.path.join(REPO_TOP_DIR, "runs")

def load_metrics(run_dir):
    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    return metrics


def plot_failure_modes(metrics_by_level, name):
    print(f'Plotting failure modes for {name}')
    # Extract failure mode data
    percentages_by_level = {}
    for level, metrics in metrics_by_level.items():
        correctness = metrics.get('correctness', {})
        failure_modes = {
            "Compilation Error": correctness["total"] - correctness["compiled"],
            "Runtime Error": correctness["runtime_error"],
            "Output Mismatch": correctness["output_mismatch"],
            "Output Shape Mismatch": correctness["output_shape_mismatch"],
            "Correct": correctness["correct"]
        }
        total = correctness["total"]
        assert total == sum(failure_modes.values()), "Total number of samples does not match the sum of failure modes"
        
        # Calculate percentages
        percentages = [v/total * 100 for v in failure_modes.values()]

        percentages_by_level[level] = percentages
    
    # Create figure and axis with more horizontal space for legend
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Create rectangle with different colored sections
    colors = ['lightcoral', 'lightsalmon', 'wheat', 'plum', 'yellowgreen']
    
    y_offset = 0.8
    for level, percentages in percentages_by_level.items():
        # Draw the main rectangle (wider and shorter)
        rect = plt.Rectangle((0, y_offset), 1, 0.2, facecolor='lightgray', edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Add level label on the left
        ax.text(-0.05, y_offset + 0.1, f'{level}', ha='center', va='center', fontweight='bold', fontsize=12)
        
        # Draw colored sections
        current_x = 0
        legend_elements = []
        for i, (label, percentage, color) in enumerate(zip(failure_modes.keys(), percentages, colors)):
            width = percentage / 100
            section = plt.Rectangle((current_x, y_offset), width, 0.2, facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(section)
            
            # Add percentage text in the center of each section
            center_x = current_x + width/2
            ax.text(center_x, y_offset + 0.1, f'{percentage:.1f}%', 
                    ha='center', va='center', fontweight='bold', fontsize=10)
            
            # Create legend element
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black'))
            
            current_x += width

        y_offset -= 0.25
    
    # Set axis properties
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add title
    plt.title(f'Failure Mode Distribution: {name}', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend outside the plot
    legend = ax.legend(legend_elements, failure_modes.keys(), 
                      loc='center left', bbox_to_anchor=(1.02, 0.5),
                      title='Failure Modes', fontsize=10)
    legend.get_title().set_fontsize(12) 
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_failure_modes.png"), bbox_inches='tight')
    plt.close()


def plot_fast_p_scores_across_p(metrics, name):
    """
    Plots the fast_p scores across different p values. Legend by number of samples or by methods(?) or levels?
    """
    print(f'Plotting fast_p scores across p for {name}')
    plt.figure(figsize=(10, 5))

    for label, metric in metrics.items():
        fast_p_scores = metric["speedups"]["torch"]["fast_p_results"]
        plt.plot(fast_p_scores.keys(), fast_p_scores.values(), marker='o', label=label)
        
    plt.ylim(0, 1.05)
    plt.xlabel('Threshold p')
    plt.ylabel('fast_p score')
    plt.title(f'Fast P Score Distribution: {name}')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_fast_p_scores.png"), bbox_inches='tight')
    plt.close() 


def plot_fast_p_by_num_samples(metrics_by_label, p="1.0", name=None):
    """
    Plots the fast_p score (for given p) by number of samples. Legend by level or methods?
    """
    print(f'Plotting fast_p by number of samples for {name} with p={p}')
    plt.figure(figsize=(10, 5))
    max_sample = 0
    for label, metrics in metrics_by_label.items():
        num_samples = list(map(lambda x: int(x) + 1, list(metrics.keys())))
        fast_p_scores = list(map(lambda x: x["speedups"]["torch"]["fast_p_results"][p] if p != "mean" else x["speedups"]["torch"]["mean_speedup_correct"], metrics.values()))

        plt.plot(num_samples, fast_p_scores, marker='o', label=label)
        max_sample = max(max_sample, max(num_samples))

    if p != "mean":
        plt.ylim(0, 1.05)

    plt.xticks(range(1, max_sample + 1))
    plt.xlabel('Number of Samples')
    plt.ylabel("Correctness" if p == "0.0" else f'Fast_{p} Score' if p != "mean" else "Mean Speedup")
    plt.legend()
    plt.title(f"Correctness by Number of Samples: {name}" if p == "0.0" else f'Fast_{p} Score by Number of Samples: {name}' if p != "mean" else f"Mean Speedup by Number of Samples: {name}")
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_correctness_by_num_samples.png" if p == "0.0" else f"{name}_fast_{p}_by_num_samples.png" if p != "mean" else f"{name}_mean_speedup_by_num_samples.png"), bbox_inches='tight')
    plt.close()

def plot_fast_p_barchart(metrics_by_label, p="1.0", name=None):
    """
    Plots the fast_p score (for given p) by number of samples. Legend by level or methods?
    """
    print(f'Plotting fast_p by number of samples for {name} with p={p}')
    plt.figure(figsize=(10, 5))

    for label, metrics in metrics_by_label.items():
        if p == "mean":
            fast_p_score = metrics["speedups"]["torch"]["mean_speedup_correct"]
        else:
            fast_p_score = metrics["speedups"]["torch"]["fast_p_results"][p]

        plt.bar(label, fast_p_score, label=label)
        plt.text(label, fast_p_score + 0.01, f'{fast_p_score:.3f}', 
                ha='center', va='bottom')


    if p != "mean":
        plt.ylim(0, 1.05)

    plt.xlabel('Method')
    plt.ylabel("Correctness" if p == "0.0" else f'Fast_{p} Score' if p != "mean" else "Mean Speedup")
    plt.title(f"Correctness by Method: {name}" if p == "0.0" else f'Fast_{p} Score by Method: {name}' if p != "mean" else f"Mean Speedup by Method: {name}")
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_correctness_by_method.png" if p == "0.0" else f"{name}_fast_{p}_by_method.png" if p != "mean" else f"{name}_mean_speedup_by_method.png"), bbox_inches='tight')
    plt.close()

def main():
    # Code to get failure modes
    parser = ArgumentParser()
    parser.add_argument("--method", type=str, default=None, help="Method to plot")
    # OR
    parser.add_argument("--level", type=int, default=None, help="Level to plot")
    parser.add_argument("--methods", type=str, default=None, help="Methods to plot")
    args = parser.parse_args()

    if args.method is None:
        if args.level is None or args.methods is None:
            raise ValueError("Either run_dir or level and methods must be provided")
        # analyze by method 
        name = f"Level_{args.level}"
        print(f'Analyzing {name} across methods')
        methods = args.methods.split(",")

        metrics_by_method_best = {}
        metrics_by_method_by_sample = {}
        for method in methods:
            run_dir = os.path.join(RUNS_DIR, method + f"_level{args.level}")
            if not os.path.exists(os.path.join(run_dir, "metrics.json")):
                print(f'Run directory {run_dir} does not exist or does not have metrics.json')
                continue
            metrics = load_metrics(run_dir)
            metrics = metrics["best_by_sample"] if "best_by_sample" in metrics else metrics
            if "0" in metrics:
                metrics_by_method_best[method] = metrics[str(max(list(map(int, metrics.keys()))))]
                metrics_by_method_by_sample[method] = metrics
            else:
                metrics_by_method_best[method] = metrics

        plot_fast_p_scores_across_p(metrics_by_method_best, name)
        plot_fast_p_barchart(metrics_by_method_best, p="mean", name=name)
        plot_fast_p_barchart(metrics_by_method_best, p="0.0", name=name)
        plot_fast_p_barchart(metrics_by_method_best, p="1.0", name=name)
        if len(metrics_by_method_by_sample) > 0:
            plot_fast_p_by_num_samples(metrics_by_method_by_sample, p="mean", name=name)
            plot_fast_p_by_num_samples(metrics_by_method_by_sample, p="0.0", name=name)
            plot_fast_p_by_num_samples(metrics_by_method_by_sample, p="1.0", name=name)

    else: # Analyze across level
        name = args.method
        print(f'Analyzing {name} across levels')
        metrics_by_level_best = {}
        metrics_by_level_by_sample = {}
        for level in [1, 2, 3, 5]:
            run_dir = os.path.join(RUNS_DIR, args.method + f"_level{level}")
            if not os.path.exists(os.path.join(run_dir, "metrics.json")):
                print(f'Run directory {run_dir} does not exist')
                continue
            metrics = load_metrics(run_dir)
            metrics = metrics["best_by_sample"] if "best_by_sample" in metrics else metrics
            if "0" in metrics:
                plot_fast_p_scores_across_p({f"n={int(k)+1}": v for k,v in metrics.items()}, f"{name}_level{level}")
                metrics_by_level_best[f"Level {level}"] = metrics[str(max(list(map(int, metrics.keys()))))]
                metrics_by_level_by_sample[f"Level {level}"] = metrics
            else:
                metrics_by_level_best[f"Level {level}"] = metrics

        plot_failure_modes(metrics_by_level_best, name)
        plot_fast_p_scores_across_p(metrics_by_level_best, name)
        if len(metrics_by_level_by_sample) > 0:
            plot_fast_p_by_num_samples(metrics_by_level_by_sample, p="mean", name=name)
            plot_fast_p_by_num_samples(metrics_by_level_by_sample, p="0.0", name=name)
            plot_fast_p_by_num_samples(metrics_by_level_by_sample, p="1.0", name=name)
        

if __name__ == "__main__":
    main()

