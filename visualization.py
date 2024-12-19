import json
import matplotlib.pyplot as plt
from typing import Literal, Dict, List


def plot_metric(
    metric: Literal['belebele', 'bbh', 'ppl', 'factoid_qa', 'mmlu'], 
    data: Dict[str, List[dict]], 
    pipelines: list | None = ['prune', 'finetune_prune', 'prune_finetune', 'iter_pf', 'iter_fp']
):
    """
    Plot the given metric for different pipelines, and save the figure to the file in plots directory with metric name as the filename.

    Args:
        metric (str): the metric to visualize.
        data (dict): the results of these experiments.
        pipelines (list): the pipelines to compare.
    """
    plt.figure()

    for pipeline in pipelines:
        values = [ex[metric] for ex in data[pipeline]]
        spars = [ex["sparsity_latest"] for ex in data[pipeline]]
        plt.plot(spars, values, marker='o', linestyle='-', label = pipeline)

    # Adding labels and title
    plt.xlabel('Sparsity', fontsize=14)  # Increase font size for x-axis label
    plt.ylabel(metric, fontsize=14)  # Increase font size for y-axis label
    plt.xticks(fontsize=12)  # Increase font size for x-axis tick labels
    plt.yticks(fontsize=12)  # Increase font size for y-axis tick labels
    plt.legend(fontsize=12)

    # Display plot
    plt.grid(True)
    plt.savefig(f'plots/{metric}.png')


def plot_metrics(
    metrics: list | None = ['belebele', 'bbh', 'ppl', 'factoid_qa', 'mmlu'], 
    results_path: str | None = 'results.json'
):
    """
    Plot metrics for different pipelines. For each metric, an image will be produced and saved in the plots directory.

    Args:
        metrics (list): list of metrics to visualize.
        results_path (str): path of the results file.
    """
    with open(results_path, 'r') as f:
        data = json.load(f)
    for metric in metrics:
        plot_metric(metric=metric, data=data)


if __name__ == '__main__':
    plot_metrics()
    