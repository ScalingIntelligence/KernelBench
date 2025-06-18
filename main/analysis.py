from argparse import ArgumentParser
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb


def load_metrics(run_dir):
    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    return metrics


def plot_failure_modes(metrics):
    pass

def plot_fast_p_scores(metrics):
    pass

def plot_fast_p_vs_num_samples(metrics):
    pass







def main():
    parser = ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    metrics = load_metrics(args.run_dir)
    
    














