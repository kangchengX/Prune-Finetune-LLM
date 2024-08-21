import matplotlib.pyplot as plt
import json
from typing import Literal


def comparisons_plot_advanced_methods(metric: Literal['belebele', 'bbh', 'ppl', 'factoid_qa', 'mmlu'], results_path: str | None = 'res.json'):
    """
    Plot metrics for different pipelines with different sparsity. Plots will be save in plots dir.

    Args:
        metric (str): metric to plot.
    """
    assert(metric in ['belebele', 'bbh', 'ppl', 'factoid_qa', 'mmlu'])
    with open(results_path, 'r') as f:
        data = json.load(f)
    if metric == 'belebele':
        values_iter= [data['iter'][i]['belebele'] for i in range(len(data['iter']))]
        pr_ft_only_iter = [values_iter[i] for i in range(len(values_iter)) if i % 2 == 1]

        spars_values_iter = [data['iter'][i]['sparsity_latest'] for i in range(len(data['iter']))]
        spars_pr_ft_only_iter = [spars_values_iter[i] for i in range(len(spars_values_iter)) if i % 2 == 1]

        values_ft_pr= [data['finetune_prune'][i]['belebele'] for i in range(len(data['finetune_prune']))]
        spars_values_ft_pr = [data['finetune_prune'][i]['sparsity_latest'] for i in range(len(data['finetune_prune']))]

        values_pr_ft = [data['prune_finetune'][i]['belebele'] for i in range(len(data['prune_finetune']))]
        spars_values_pr_ft = [data['prune_finetune'][i]['sparsity_latest'] for i in range(len(data['prune_finetune']))]
        values_prune= [data['prune'][i]['belebele'] for i in range(len(data['prune']))]
        spars_values_prune = [data['prune'][i]['sparsity_latest'] for i in range(len(data['prune']))]
        #plt.plot(range(spars_values_iter), belebele_values_iter, marker='o', linestyle='-', label = 'iter')

        plt.figure()
        plt.plot(spars_pr_ft_only_iter, pr_ft_only_iter, marker='o', linestyle='-', label = 'iter')
        plt.plot(spars_values_pr_ft, values_pr_ft, marker='o', linestyle='-', label = 'prune_finetune')
        plt.plot(spars_values_ft_pr, values_ft_pr, marker='o', linestyle='-', label = 'finetune_prune')
        plt.plot(spars_values_prune, values_prune, marker='o', linestyle='-', label = 'prune')
        # Adding labels and title
        plt.xlabel('Sparsity', fontsize=14)  # Increase font size for x-axis label
        plt.ylabel('Belebele', fontsize=14)  # Increase font size for y-axis label
        plt.xticks(fontsize=12)  # Increase font size for x-axis tick labels
        plt.yticks(fontsize=12)  # Increase font size for y-axis tick labels
        plt.legend(fontsize=12) 
        plt.axhline(y=25, color='red', linestyle='--')
        plt.legend()
        # plt.title('Belebele Performance vs Sparsity')

        # Display plot
        plt.grid(True)
        plt.savefig('plots/belebele_comp_advanced.png')
        plt.show()
        

    elif metric == 'bbh':
        values_iter= [data['iter'][i]['bbh'] for i in range(len(data['iter']))]
        pr_ft_only_iter = [values_iter[i] for i in range(len(values_iter)) if i % 2 == 1]

        spars_values_iter = [data['iter'][i]['sparsity_latest'] for i in range(len(data['iter']))]
        spars_pr_ft_only_iter = [spars_values_iter[i] for i in range(len(spars_values_iter)) if i % 2 == 1]

        values_pr_ft = [data['prune_finetune'][i]['bbh'] for i in range(len(data['prune_finetune']))]
        spars_values_pr_ft = [data['prune_finetune'][i]['sparsity_latest'] for i in range(len(data['prune_finetune']))]
        values_prune= [data['prune'][i]['bbh'] for i in range(len(data['prune']))]
        spars_values_prune = [data['prune'][i]['sparsity_latest'] for i in range(len(data['prune']))]

        values_ft_pr= [data['finetune_prune'][i]['bbh'] for i in range(len(data['finetune_prune']))]
        spars_values_ft_pr = [data['finetune_prune'][i]['sparsity_latest'] for i in range(len(data['finetune_prune']))]
        #plt.plot(range(spars_values_iter), belebele_values_iter, marker='o', linestyle='-', label = 'iter')
        
        plt.figure()
        plt.plot(spars_pr_ft_only_iter, pr_ft_only_iter, marker='o', linestyle='-', label = 'iter')
        plt.plot(spars_values_pr_ft, values_pr_ft, marker='o', linestyle='-', label = 'prune_finetune')
        plt.plot(spars_values_ft_pr, values_ft_pr, marker='o', linestyle='-', label = 'finetune_prune')
        plt.plot(spars_values_prune, values_prune, marker='o', linestyle='-', label = 'prune')
        plt.axhline(y=20, color='red', linestyle='--')
        # Adding labels and title
        plt.xlabel('Sparsity', fontsize=14)  # Increase font size for x-axis label
        plt.ylabel('BBH', fontsize=14)  # Increase font size for y-axis label
        plt.xticks(fontsize=12)  # Increase font size for x-axis tick labels
        plt.yticks(fontsize=12)  # Increase font size for y-axis tick labels
        plt.legend(fontsize=12) 
        plt.legend()
        # plt.title('BBH Performance vs Sparsity')

        # Display plot
        plt.grid(True)
        plt.show()
        plt.savefig('plots/bbh_comp_advanced.png')

    elif metric == 'ppl':
        values_iter= [data['iter'][i]['ppl'] for i in range(len(data['iter']))]
        pr_ft_only_iter = [values_iter[i] for i in range(len(values_iter)) if i % 2 == 1]

        spars_values_iter = [data['iter'][i]['sparsity_latest'] for i in range(len(data['iter']))]
        spars_pr_ft_only_iter = [spars_values_iter[i] for i in range(len(spars_values_iter)) if i % 2 == 1]
        values_pr_ft = [data['prune_finetune'][i]['ppl'] for i in range(len(data['prune_finetune']))]
        spars_values_pr_ft = [data['prune_finetune'][i]['sparsity_latest'] for i in range(len(data['prune_finetune']))]

        values_ft_pr= [data['finetune_prune'][i]['ppl'] for i in range(len(data['finetune_prune']))]
        spars_values_ft_pr = [data['finetune_prune'][i]['sparsity_latest'] for i in range(len(data['finetune_prune']))]

        values_prune= [data['prune'][i]['ppl'] for i in range(len(data['prune']))]
        spars_values_prune = [data['prune'][i]['sparsity_latest'] for i in range(len(data['prune']))]
        #plt.plot(range(spars_values_iter), belebele_values_iter, marker='o', linestyle='-', label = 'iter')

        plt.figure()
        plt.plot(spars_pr_ft_only_iter, pr_ft_only_iter, marker='o', linestyle='-', label = 'iter')
        plt.plot(spars_values_pr_ft, values_pr_ft, marker='o', linestyle='-', label = 'prune_finetune')
        plt.plot(spars_values_ft_pr[:-1], values_ft_pr[:-1], marker='o', linestyle='-', label = 'finetune_prune')
        plt.plot(spars_values_prune[:-1], values_prune[:-1], marker='o', linestyle='-', label = 'prune')
        # Adding labels and title
        plt.xlabel('Sparsity', fontsize=14)  # Increase font size for x-axis label
        plt.ylabel('PPL', fontsize=14)  # Increase font size for y-axis label
        plt.xticks(fontsize=12)  # Increase font size for x-axis tick labels
        plt.yticks(fontsize=12)  # Increase font size for y-axis tick labels
        plt.legend(fontsize=12) 
        plt.legend()
        # plt.title('Perplexity Performance vs Sparsity')

        # Display plot
        plt.grid(True)
        plt.show()
        plt.savefig('plots/ppl_comp_advanced.png')

    elif metric == 'factoid_qa':
        values_iter= [data['iter'][i]['factoid_qa'] for i in range(len(data['iter']))]
        pr_ft_only_iter = [100*values_iter[i] for i in range(len(values_iter)) if i % 2 == 1]

        spars_values_iter = [data['iter'][i]['sparsity_latest'] for i in range(len(data['iter']))]
        spars_pr_ft_only_iter = [spars_values_iter[i] for i in range(len(spars_values_iter)) if i % 2 == 1]
        values_pr_ft = [100*data['prune_finetune'][i]['factoid_qa'] for i in range(len(data['prune_finetune']))]
        spars_values_pr_ft = [data['prune_finetune'][i]['sparsity_latest'] for i in range(len(data['prune_finetune']))]
        values_ft_pr= [100*data['finetune_prune'][i]['factoid_qa'] for i in range(len(data['finetune_prune']))]
        spars_values_ft_pr = [data['finetune_prune'][i]['sparsity_latest'] for i in range(len(data['finetune_prune']))]
        values_prune= [100*data['prune'][i]['factoid_qa'] for i in range(len(data['prune']))]
        spars_values_prune = [data['prune'][i]['sparsity_latest'] for i in range(len(data['prune']))]
        #plt.plot(range(spars_values_iter), belebele_values_iter, marker='o', linestyle='-', label = 'iter')

        plt.figure()
        plt.plot(spars_pr_ft_only_iter, pr_ft_only_iter, marker='o', linestyle='-', label = 'iter')
        plt.plot(spars_values_pr_ft, values_pr_ft, marker='o', linestyle='-', label = 'prune_finetune')
        plt.plot(spars_values_ft_pr, values_ft_pr, marker='o', linestyle='-', label = 'finetune_prune')
        plt.plot(spars_values_prune[:-1], values_prune[:-1], marker='o', linestyle='-', label = 'prune')
        # plt.axhline(y=50, color='red', linestyle='--')
        # Adding labels and title
        plt.xlabel('Sparsity', fontsize=14)  # Increase font size for x-axis label
        plt.ylabel('Factoid_qa', fontsize=14)  # Increase font size for y-axis label
        plt.xticks(fontsize=12)  # Increase font size for x-axis tick labels
        plt.yticks(fontsize=12)  # Increase font size for y-axis tick labels
        plt.legend(fontsize=12) 
        plt.legend()
        # plt.title('Factoid_qa Performance vs Sparsity')

        # Display plot
        plt.grid(True)
        plt.show()
        plt.savefig('plots/factoid_qa_comp_advanced.png')
    
    elif metric == 'mmlu':
        values_iter= [data['iter'][i]['mmlu'] for i in range(len(data['iter']))]
        pr_ft_only_iter = [values_iter[i] for i in range(len(values_iter)) if i % 2 == 1]

        spars_values_iter = [data['iter'][i]['sparsity_latest'] for i in range(len(data['iter']))]
        spars_pr_ft_only_iter = [spars_values_iter[i] for i in range(len(spars_values_iter)) if i % 2 == 1]
        values_pr_ft = [data['prune_finetune'][i]['mmlu'] for i in range(len(data['prune_finetune']))]
        spars_values_pr_ft = [data['prune_finetune'][i]['sparsity_latest'] for i in range(len(data['prune_finetune']))]
        values_ft_pr= [data['finetune_prune'][i]['mmlu'] for i in range(len(data['finetune_prune']))]
        spars_values_ft_pr = [data['finetune_prune'][i]['sparsity_latest'] for i in range(len(data['finetune_prune']))]
        values_prune= [data['prune'][i]['mmlu'] for i in range(len(data['prune']))]
        spars_values_prune = [data['prune'][i]['sparsity_latest'] for i in range(len(data['prune']))]
        #plt.plot(range(spars_values_iter), belebele_values_iter, marker='o', linestyle='-', label = 'iter')

        plt.figure()
        plt.plot(spars_pr_ft_only_iter, pr_ft_only_iter, marker='o', linestyle='-', label = 'iter')
        plt.plot(spars_values_pr_ft, values_pr_ft, marker='o', linestyle='-', label = 'prune_finetune')
        plt.plot(spars_values_ft_pr, values_ft_pr, marker='o', linestyle='-', label = 'finetune_prune')
        plt.plot(spars_values_prune[:-1], values_prune[:-1], marker='o', linestyle='-', label = 'prune')
        plt.axhline(y=25, color='red', linestyle='--')
        # Adding labels and title
        plt.xlabel('Sparsity', fontsize=14)  # Increase font size for x-axis label
        plt.ylabel('MMLU', fontsize=14)  # Increase font size for y-axis label
        plt.xticks(fontsize=12)  # Increase font size for x-axis tick labels
        plt.yticks(fontsize=12)  # Increase font size for y-axis tick labels
        plt.legend(fontsize=12) 
        plt.legend()
        # plt.title('MMLU Performance vs Sparsity')

        # Display plot
        plt.grid(True)
        plt.show()
        plt.savefig('plots/mmlu_comp_advanced.png')

    else:
        raise ValueError(f'Unsupported metric :{metric}')


if __name__ == '__main__':
    metrics = ['belebele', 'bbh', 'ppl', 'factoid_qa', 'mmlu']
    for m in metrics:
        comparisons_plot_advanced_methods(m)

    