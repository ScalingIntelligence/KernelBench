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


MODEL_TO_NAME = {
    "deepseek_r1": "DeepSeek-R1",
    "QwQ_32B": "QwQ-32B",
    "qwen_2.5_7b": "Qwen2.5-7B-Instruct",
    "qwen_2.5_1.5b": "Qwen2.5-1.5B-Instruct",
}

METHOD_TO_NAME = {
    "base": "Base",
    "best_of_n": "Best-of-N",
    "IR": "Iterative Refinement",
    "metr": "METR",
}


def to_run_dir(method, level, model):
    return os.path.join(RUNS_DIR, f"{method}_level{level}_{model}")


def plot_failure_modes(metrics_by_label, name, plot_dir):
    print(f'Plotting failure modes for {name}')
    # Extract failure mode data
    percentages_by_label = {}
    for label, metrics in metrics_by_label.items():
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

        percentages_by_label[label] = percentages
    
    # Create figure and axis with more horizontal space for legend
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Create rectangle with different colored sections
    colors = ['lightcoral', 'lightsalmon', 'wheat', 'plum', 'yellowgreen']
    
    num_labels = len(percentages_by_label)
    height_per_rectangle = 0.20
    spacing_between_rectangles = 0.05
    y_offset = (num_labels - 1) * (height_per_rectangle + spacing_between_rectangles) # top most rectangle
    legend_elements = []
    for label, percentages in percentages_by_label.items():
        # Draw the main rectangle (wider and shorter)
        rect = plt.Rectangle((0, y_offset), 1, height_per_rectangle, facecolor='lightgray', edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Add level label on the left
        ax.text(-0.05, y_offset + height_per_rectangle/2, f'{label}', ha='center', va='center', fontweight='bold', fontsize=12)
        
        # Draw colored sections
        legend_elements = []
        current_x = 0
        for i, (label, percentage, color) in enumerate(zip(failure_modes.keys(), percentages, colors)):
            width = percentage / 100
            section = plt.Rectangle((current_x, y_offset), width, height_per_rectangle, facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(section)
            
            # Add percentage text in the center of each section
            center_x = current_x + width/2
            ax.text(center_x, y_offset + height_per_rectangle/2, f'{percentage:.1f}%', 
                    ha='center', va='center', fontweight='bold', fontsize=10)
            
            # Create legend element
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black'))
            
            current_x += width

        y_offset -= height_per_rectangle + spacing_between_rectangles
    
    # Set axis properties
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, num_labels * (height_per_rectangle + spacing_between_rectangles) - spacing_between_rectangles)
    ax.axis('off')
    
    # Add title
    plt.title(f'Failure Mode Distribution: {name}', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend outside the plot
    legend = ax.legend(legend_elements, failure_modes.keys(), 
                      loc='center left', bbox_to_anchor=(1.02, 0.5),
                      title='Failure Modes', fontsize=10)
    legend.get_title().set_fontsize(12) 
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "failure_modes.png"), bbox_inches='tight')
    plt.close()


def plot_fast_p_scores_across_p(metrics_by_label, name, plot_dir):
    """
    Plots the fast_p scores across different p values. 
    """
    print(f'Plotting fast_p scores across p for {name}')
    plt.figure(figsize=(10, 5))

    for label, metrics in metrics_by_label.items():
        fast_p_scores = metrics["speedups"]["torch"]["fast_p_results"]
        plt.plot(fast_p_scores.keys(), fast_p_scores.values(), marker='o', label=label)
        
    plt.ylim(0, 1.05)
    plt.xlabel('Threshold p')
    plt.ylabel('fast_p score')
    plt.title(f'Fast P Score Distribution: {name}')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "fast_p_scores.png"), bbox_inches='tight')
    plt.close() 


def plot_fast_p_by_num_samples(metrics_by_label_by_sample, p="1.0", name=None, plot_dir=None):
    """
    Plots the fast_p score (for given p) by number of samples. 
    Values of p: mean, 0.0 (correctness), 0.5, 1.0, 2.0, etc
    """
    print(f'Plotting fast_p by number of samples for {name} with p={p}')
    plt.figure(figsize=(10, 5))

    max_sample = 0
    for label, metrics in metrics_by_label_by_sample.items():
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
    plt.savefig(os.path.join(plot_dir, f"correctness_by_num_samples.png" if p == "0.0" else f"fast_{p}_by_num_samples.png" if p != "mean" else f"mean_speedup_by_num_samples.png"), bbox_inches='tight')
    plt.close()


def plot_fast_p_by_epochs(metrics_by_epoch, p="1.0", name=None, plot_dir=None):
    plt.figure(figsize=(10, 5))

    for label, metrics in metrics_by_epoch.items():
        epochs = list(metrics.keys())
        fast_p_scores = list(map(lambda x: x["speedups"]["torch"]["fast_p_results"][p] if p != "mean" else x["speedups"]["torch"]["mean_speedup_correct"], metrics.values()))

        plt.plot(epochs, fast_p_scores, marker='o', label=label)

    if p != "mean":
        plt.ylim(0, 1.05)

    plt.xlabel('Number of Epochs')
    plt.ylabel("Correctness" if p == "0.0" else f'Fast_{p} Score' if p != "mean" else "Mean Speedup")
    plt.legend()
    plt.title(f"Correctness by Number of Epochs: {name}" if p == "0.0" else f'Fast_{p} Score by Number of Epochs: {name}' if p != "mean" else f"Mean Speedup by Number of Epochs: {name}")
    plt.savefig(os.path.join(plot_dir, f"correctness_by_num_epochs.png" if p == "0.0" else f"fast_{p}_by_num_epochs.png" if p != "mean" else f"mean_speedup_by_num_epochs.png"), bbox_inches='tight')
    plt.close()


def plot_fast_p_barchart(metrics_by_label, p="1.0", axis="Method", name=None, plot_dir=None):
    """
    Plots the fast_p score (for given p)
    Values of p: mean, 0.0 (correctness), 0.5, 1.0, 2.0, etc
    """
    print(f'Plotting fast_p barchart for {name} with p={p}')
    plt.figure(figsize=(10, 5))

    for label, metrics in metrics_by_label.items():
        if p == "mean":
            fast_p_score = metrics["speedups"]["torch"]["mean_speedup_correct"]
        else:
            fast_p_score = metrics["speedups"]["torch"]["fast_p_results"][p]

        plt.bar(label, fast_p_score, label=label)
        plt.text(label, fast_p_score + 0.01, f'{fast_p_score:.2f}', 
                ha='center', va='bottom')

    if p != "mean":
        plt.ylim(0, 1.05)

    plt.xlabel(axis)
    plt.ylabel("Correctness" if p == "0.0" else f'Fast_{p} Score' if p != "mean" else "Mean Speedup")
    plt.title(f"Correctness by {axis}: {name}" if p == "0.0" else f'Fast_{p} Score by {axis}: {name}' if p != "mean" else f"Mean Speedup by {axis}: {name}")
    plt.savefig(os.path.join(plot_dir, f"correctness.png" if p == "0.0" else f"fast_{p}.png" if p != "mean" else f"mean_speedup.png"), bbox_inches='tight')
    plt.close()


def plot_everything(metrics_by_label, metrics_by_label_by_sample, axis, name, plot_dir):
    plot_failure_modes(metrics_by_label, name, plot_dir)
    plot_fast_p_scores_across_p(metrics_by_label, name=name, plot_dir=plot_dir)
    plot_fast_p_barchart(metrics_by_label, p="mean", axis=axis, name=name, plot_dir=plot_dir) # Mean speedup
    plot_fast_p_barchart(metrics_by_label, p="0.0", axis=axis, name=name, plot_dir=plot_dir) # Correctness
    plot_fast_p_barchart(metrics_by_label, p="1.0", axis=axis, name=name, plot_dir=plot_dir) # Fast_1.0 score
    if len(metrics_by_label_by_sample) > 0: # for test-time scaling methods: plot across number of samples
        plot_fast_p_by_num_samples(metrics_by_label_by_sample, p="mean", name=name, plot_dir=plot_dir) # Mean speedup
        plot_fast_p_by_num_samples(metrics_by_label_by_sample, p="0.0", name=name, plot_dir=plot_dir) # Correctness
        plot_fast_p_by_num_samples(metrics_by_label_by_sample, p="1.0", name=name, plot_dir=plot_dir) # Fast_1.0 score


def main():
    # Code to get failure modes
    parser = ArgumentParser()
    parser.add_argument("--axis", type=str, choices=["method", "level", "model"], required=True, help="Axis to plot")
    parser.add_argument("--method", type=str, default=None, help="Method to plot")
    parser.add_argument("--methods", type=str, default=None, help="Methods to plot")
    parser.add_argument("--level", type=int, default=None, help="Level to plot")
    parser.add_argument("--levels", type=str, default=None, help="Levels to plot")
    parser.add_argument("--model", type=str, default=None, help="Model to plot")
    parser.add_argument("--models", type=str, default=None, help="Models to plot")
    parser.add_argument("--tag", type=str, default=None, help="Tag to plot")
    args = parser.parse_args()

    # Initialize 
    name = None
    plot_dir = None
    metrics_by_label = {} # dict of label -> metrics
    metrics_by_label_by_sample = {} # dict of label -> num_samples -> metrics

    if args.axis == "method": # Analyze across methods of a given level and model
        assert args.levels is None and args.models is None, "Cannot analyze across methods and levels/models"
        assert args.level is not None and args.model is not None, "Specify a level and model"

        if args.methods is None:
            methods = ["base", "best_of_n", "IR", "metr"]
        else:
            methods = args.methods.split(",")
        print(f'Analyzing results Level: {args.level} Model: {args.model} across Methods: {methods}')
        name = f"Level {args.level} ({args.model})"
        plot_dir = os.path.join(PLOT_DIR, "across_methods", f"level{args.level}_{args.model}" + (f"_{args.tag}" if args.tag else ""))

        for method in methods:
            method_name = METHOD_TO_NAME[method]
            run_dir = to_run_dir(method, args.level, args.model)
            if not os.path.exists(os.path.join(run_dir, "metrics.json")):
                print(f'Run directory {run_dir} does not exist or does not have metrics.json')
                continue

            metrics = load_metrics(run_dir)
            metrics = metrics["best_by_sample"] if "best_by_sample" in metrics else metrics
            if "0" in metrics: # for test-time scaling methods
                metrics_by_label[method_name] = metrics[str(max(list(map(int, metrics.keys()))))]
                metrics_by_label_by_sample[method_name] = metrics
            else: # for base
                metrics_by_label[method_name] = metrics

    elif args.axis == "level": # Analyze across level of given method and model
        assert args.methods is None and args.models is None, "Cannot analyze across levels and methods/models"
        assert args.method is not None and args.model is not None, "Specify a method and model"

        if args.levels is None:
            levels = [1, 2, 3, 5]
        else:
            levels = list(map(int, args.levels.split(",")))

        print(f'Analyzing results Method: {args.method} Model: {args.model} across levels: {levels}')
        name = f"{METHOD_TO_NAME[args.method]} ({args.model})"
        plot_dir = os.path.join(PLOT_DIR, "across_levels", f"{args.method}_{args.model}" + (f"_{args.tag}" if args.tag else ""))

        for level in levels:
            level_name = f"Level {level}"
            run_dir = to_run_dir(args.method, level, args.model)
            if not os.path.exists(os.path.join(run_dir, "metrics.json")):
                print(f'Run directory {run_dir} does not exist or does not have metrics.json')
                continue

            metrics = load_metrics(run_dir)
            metrics = metrics["best_by_sample"] if "best_by_sample" in metrics else metrics
            if "0" in metrics:
                metrics_by_label[level_name] = metrics[str(max(list(map(int, metrics.keys()))))]
                metrics_by_label_by_sample[level_name] = metrics
            else:
                metrics_by_label[level_name] = metrics

    elif args.axis == "model": # Analyze across model of given method and level
        assert args.methods is None and args.levels is None, "Cannot analyze across models and methods/levels"
        assert args.method is not None and args.level is not None, "Specify a method and level"

        if args.models is None:
            models = ["QwQ-32B", "Qwen2.5-7B-Instruct", "Qwen2.5-1.5B-Instruct", "DeepSeek-R1"]
        else:
            models = args.models.split(",")

        print(f'Analyzing results Method: {args.method} Level: {args.level} across models: {models}')
        name = f"{METHOD_TO_NAME[args.method]} (Level {args.level})"
        plot_dir = os.path.join(PLOT_DIR, "across_models", f"{args.method}_level{args.level}" + (f"_{args.tag}" if args.tag else ""))

        for model in models:
            model_name = model
            run_dir = to_run_dir(args.method, args.level, model)
            if not os.path.exists(os.path.join(run_dir, "metrics.json")):
                print(f'Run directory {run_dir} does not exist or does not have metrics.json')
                continue

            metrics = load_metrics(run_dir)
            metrics = metrics["best_by_sample"] if "best_by_sample" in metrics else metrics
            if "0" in metrics:
                metrics_by_label[model_name] = metrics[str(max(list(map(int, metrics.keys()))))]
                metrics_by_label_by_sample[model_name] = metrics
            else:
                metrics_by_label[model_name] = metrics

    # Plot everything
    os.makedirs(plot_dir, exist_ok=True)
    plot_everything(metrics_by_label, metrics_by_label_by_sample, args.axis, name, plot_dir)


def sft_analysis():
    epochs = [4, 10, 20, 30, 40, 50]

    metrics_by_level = {}

    for level in [1, 2]:
        metrics_by_epoch = {}

        run_dir = os.path.join(RUNS_DIR, f"base_level{level}_Qwen2.5-7B-Instruct")
        metrics = load_metrics(run_dir)
        metrics_by_epoch[0] = metrics

        for epoch in epochs:
            run_dir = os.path.join(RUNS_DIR, f"base_level{level}_Qwen2.5-7B-Instruct-SFT1-{epoch}")
            if not os.path.exists(os.path.join(run_dir, "metrics.json")):
                print(f'Run directory {run_dir} does not exist or does not have metrics.json')
                continue
            metrics = load_metrics(run_dir)
            metrics_by_epoch[epoch] = metrics

        metrics_by_level[f"Level {level}"] = metrics_by_epoch
    
    plot_dir = os.path.join(PLOT_DIR, "sft_analysis", "SFT1")
    os.makedirs(plot_dir, exist_ok=True)
    plot_fast_p_by_epochs(metrics_by_level, p="0.0", name="SFT1", plot_dir=plot_dir)
    plot_fast_p_by_epochs(metrics_by_level, p="1.0", name="SFT1", plot_dir=plot_dir)




if __name__ == "__main__":
    # main()
    sft_analysis()

