import os
cache_path = './hf_cache/'

os.environ['HF_HOME']=cache_path
os.environ['TRANSFORMERS_CACHE'] = cache_path

import torch
import matplotlib.pyplot as plt
import numpy as np

from utils import get_llm, find_layers
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import AutoModelForCausalLM,AutoTokenizer


def plot_layers_distr(model, plot_name, file_name):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    if isinstance(model, PeftModelForCausalLM):
        layers = model.model.model.layers
    else:
        layers = model.model.layers

    keys = ['self_attn.q_proj.lora_A.default', 'self_attn.q_proj.lora_B.default', 'self_attn.v_proj.lora_A.default', 'self_attn.v_proj.lora_B.default']
    means = {key: [] for key in keys}
    stddevs = {key: [] for key in keys}

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        subset_lora = {
            key: value
            for key, value in subset.items()
            if 'lora' in key
        }
        assert len(subset_lora) == 4

        for name in subset_lora:
            W = subset_lora[name].weight.data
            means[name].append(W.mean().item())
            stddevs[name].append(W.std().item())

    fig, axs = plt.subplots(4, 2, figsize=(6, 7.2))
    fig.suptitle(plot_name)

    for i, key in enumerate(keys):
        ax_mean = axs[i][0]
        ax_stddev = axs[i][1]

        ax_mean.bar(range(len(means[key])), means[key], color='blue')
        ax_mean.set_title(f'Mean for {key}')
        ax_mean.set_xlabel('Layer')
        ax_mean.set_ylabel('Mean')

        ax_stddev.bar(range(len(stddevs[key])), stddevs[key], color='orange')
        ax_stddev.set_title(f'Standard Deviation for {key}')
        ax_stddev.set_xlabel('Layer')
        ax_stddev.set_ylabel('Standard Deviation')

    plt.tight_layout()
    plt.savefig("plots/" + file_name)

    model.config.use_cache = use_cache 

# def cal_metrics_weights(model, metrics=('mean','std')):
#     '''calculate the metrics of weights per layer'''
#     use_cache = model.config.use_cache 
#     model.config.use_cache = False 

#     layers = model.model.layers
#     attn_std = []
#     attn_mean = []
#     mlp_std = []
#     mlp_mean = []

#     for i in range(len(layers)):
#         layer = layers[i]
#         subset = find_layers(layer)

#         # Mean and std of attn
#         attn_weights = torch.cat([(subset[name].weight.data) for name in subset][:4])
#         attn_std.append(attn_weights.std().item())
#         attn_mean.append(attn_weights.mean().item())

#         # Mean and std of MLP
#         # mlp_weights = torch.cat([(subset[name].weight.data) for name in subset][4:])
#         mlp_weights = torch.cat([(subset[name].weight.data) for name in subset][4:-1] + [torch.t(subset['mlp.down_proj'].weight.data)])
#         mlp_std.append(mlp_weights.std().item())
#         mlp_mean.append(mlp_weights.mean().item())

#     model.config.use_cache = use_cache 

#     results = {'mean':(mlp_mean,attn_mean),'std':(attn_mean,attn_std)}

#     return results


def plot_layers_outliers(model, plot_name, file_name):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    if isinstance(model, PeftModelForCausalLM):
        layers = model.model.model.layers
    else:
        layers = model.model.layers

    keys = ['self_attn.q_proj.lora_A.default', 'self_attn.q_proj.lora_B.default', 'self_attn.v_proj.lora_A.default', 'self_attn.v_proj.lora_B.default']
    percentages = {key: [] for key in keys}

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        subset_lora = {
            key: value
            for key, value in subset.items()
            if 'lora' in key
        }
        assert len(subset_lora) == 4

        for name in subset_lora:
            W = subset_lora[name].weight.data
            std = W.std().item()
            mean = W.mean().item()
            conf_interval = 1.96 * std / np.sqrt(len(W))  # 95% confidence interval
            outside_interval = np.sum(((W < mean - conf_interval) | (W > mean + conf_interval)).numpy().astype(int))
            percentage_outside = outside_interval / (W.numpy().shape[0]*W.numpy().shape[1]) * 100
            percentages[name].append(percentage_outside)

    fig, axs = plt.subplots(2, 2, figsize=(6, 4.8))
    fig.suptitle(plot_name)

    for i, key in enumerate(keys):
        ax = axs[i // 2][i % 2]
        ax.bar(range(len(percentages[key])), percentages[key], color='green')
        ax.set_title(f'Percentage outside 95% CI for {key}')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Percentage')

    plt.tight_layout()
    plt.savefig("plots/" + file_name)

    model.config.use_cache = use_cache


# def plot_outliers_compare(results1:dict, results2:dict, model_name1:str, model_name2:str):
#     metrics = list(results1.keys())
#     for metric in metrics:
#         metric_values_1_mlp = results1[metric][0]
#         metric_values_1_attn = results1[metric][1]
#         metric_values_2_mlp = results2[metric][0]
#         metric_values_2_attn = results2[metric][1]

#         fig, ax = plt.subplots(2,1,figsize = (12, 12))
#         ax[0].plot(range(1, len(metric_values_1_mlp)+1), metric_values_1_mlp,
#                    'r-^', markersize=5, markerfacecolor='none', label = 'model '+model_name1)
#         ax[0].plot(range(1, len(metric_values_2_mlp)+1), metric_values_2_mlp,
#                    'g-s',markersize=5,markerfacecolor='none', label = 'model '+model_name2)

#         ax[1].plot(range(1, len(metric_values_1_attn)+1),metric_values_1_attn,
#                    'r-^', markersize=5, markerfacecolor='none', label = 'model '+model_name1)
#         ax[1].plot(range(1, len(metric_values_2_attn)+1), metric_values_2_attn, 
#                    'g-s', markersize=5, markerfacecolor='none', label = 'model '+model_name2)

#         plt.legend()
#         plt.savefig(model_name1+'-'+model_name2+'_'+metric+'.png')


def plot_layers_min_max(model, plot_name, file_name):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    if isinstance(model, PeftModelForCausalLM):
        layers = model.model.model.layers
    else:
        layers = model.model.layers

    keys = ['self_attn.q_proj.lora_A.default', 'self_attn.q_proj.lora_B.default', 'self_attn.v_proj.lora_A.default', 'self_attn.v_proj.lora_B.default']
    min = {key: [] for key in keys}
    max = {key: [] for key in keys}

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        subset_lora = {
            key: value
            for key, value in subset.items()
            if 'lora' in key
        }
        assert len(subset_lora) == 4

        for name in subset_lora:
            W = subset_lora[name].weight.data
            min[name].append(W.min().item())
            max[name].append(W.max().item())

    fig, axs = plt.subplots(4, 2, figsize=(6, 7.2))
    fig.suptitle(plot_name)

    for i, key in enumerate(keys):
        ax_mean = axs[i][0]
        ax_stddev = axs[i][1]

        ax_mean.bar(range(len(min[key])), min[key], color='yellow')
        ax_mean.set_title(f'Minimum for {key}')
        ax_mean.set_xlabel('Layer')
        ax_mean.set_ylabel('Minimum')

        ax_stddev.bar(range(len(max[key])), max[key], color='red')
        ax_stddev.set_title(f'Maximum for {key}')
        ax_stddev.set_xlabel('Layer')
        ax_stddev.set_ylabel('Maximum')

    plt.tight_layout()
    plt.savefig("plots/" + file_name)

    model.config.use_cache = use_cache


import torch
import matplotlib.pyplot as plt

def plot_layers_cumulative_diff(model_a, model_b, name_a, name_b, file_name, top_k_percent=100):
    use_cache_a = model_a.config.use_cache
    use_cache_b = model_b.config.use_cache
    model_a.config.use_cache = False
    model_b.config.use_cache = False

    layers_a = model_a.model.layers
    layers_b = model_b.model.layers

    attn_diff = []
    mlp_diff = []

    for i in range(len(layers_a)):
        a = layers_a[i]
        subset_a = find_layers(a)
        b = layers_b[i]
        subset_b = find_layers(b)

        # Calculate the attention weights differences
        attn_weights_a = torch.cat([(subset_a[name].weight.data) for name in subset_a][:4])
        attn_weights_b = torch.cat([(subset_b[name].weight.data) for name in subset_b][:4])

        attn_diff_layer = (attn_weights_b - attn_weights_a).abs()
        attn_diff_top_k = torch.topk(attn_diff_layer.view(-1), int(len(attn_diff_layer.view(-1)) * top_k_percent / 100), sorted=False)
        attn_diff.append(attn_diff_top_k.values.mean().item())

        # Calculate the MLP weights differences
        mlp_weights_a = torch.cat([(subset_a[name].weight.data) for name in subset_a][4:-1] + [torch.t(subset_a['mlp.down_proj'].weight.data)])
        mlp_weights_b = torch.cat([(subset_b[name].weight.data) for name in subset_b][4:-1] + [torch.t(subset_b['mlp.down_proj'].weight.data)])
        
        mlp_diff_layer = (mlp_weights_b - mlp_weights_a).abs()
        mlp_diff_top_k = torch.topk(mlp_diff_layer.view(-1), int(len(mlp_diff_layer.view(-1)) * top_k_percent / 100), sorted=False)
        mlp_diff.append(mlp_diff_top_k.values.mean().item())

    plt.figure(figsize=(7.2, 3.6))
    plt.suptitle(f'Weight Absolute Difference between \n{name_a} and {name_b}', fontsize=14)  

    # Plot attention and MLP weights differences
    plt.subplot(1, 2, 1)
    plt.bar(range(len(attn_diff)), attn_diff, color='orange')
    plt.title('Attention Weights', fontsize=14)
    plt.xlabel('Layer')
    plt.ylabel('Mean Difference')

    plt.subplot(1, 2, 2)
    plt.bar(range(len(mlp_diff)), mlp_diff, color='green')
    plt.title('MLP Weights', fontsize=14)
    plt.xlabel('Layer')
    plt.ylabel('Mean Difference') 

    plt.tight_layout()
    plt.savefig("plots/" + file_name)

    # Restore the original cache configuration
    model_a.config.use_cache = use_cache_a
    model_b.config.use_cache = use_cache_b


# model_path = "baffo32/decapoda-research-llama-7B-hf"
# model_path = "models/prune_ft04"
# model_path = "models/iter/prune_ft0_77"


models = {
    'models/prune_ft05': 'Pruning 0.5 + Finetuning 0.1 epoch',
    'models/ft_iter': 'Finetuning 0.5 epoch',
    'models/ft': 'Finetuning 0.1 epoch'
    }



for path in models.keys():
    plot_name = models[path]
    file_name = path.split("/")[-1]
    # saved_model = AutoPeftModelForCausalLM.from_pretrained(path + "/adapter")
    # plot_layers_distr(saved_model, plot_name, file_name)
    #plot_layers_outliers(saved_model, plot_name, file_name+"_out")
    # plot_layers_min_max(saved_model,plot_name,file_name+"_min_max")



# path_a = "/cs/student/projects3/COMP0087/grp1/models/iter/prune_ft0_77"
# path_b = "/cs/student/projects3/COMP0087/grp1/models/prune_ft05"
# model_a = AutoModelForCausalLM.from_pretrained(path_a)
# model_b = AutoModelForCausalLM.from_pretrained(path_b)

# plot_layers_cumulative_diff(model_a, model_b,  "(Pr-Ft)x5 [sp:0.42]", "Pr-Ft [sp:0.42]", "pr-ft_042", top_k_percent=5)


# path_a = "/cs/student/projects3/COMP0087/grp1/models/iter/prune_ft0_85"
# path_b = "/cs/student/projects3/COMP0087/grp1/models/prune_033"
# model_a = AutoModelForCausalLM.from_pretrained(path_a)
# model_b = AutoModelForCausalLM.from_pretrained(path_b)

# plot_layers_cumulative_diff(model_a, model_b,  "(Pr-Ft)x4 [sp:0.33]", "Pr [sp:0.33]", "pr-ft_033_vs_pr_033", top_k_percent=5)

# path_a = "/cs/student/projects3/COMP0087/grp1/models/iter/prune_ft0_10"
# path_b = "/cs/student/projects3/COMP0087/grp1/models/prune_ft01"
# model_a = AutoModelForCausalLM.from_pretrained(path_a)
# model_b = AutoModelForCausalLM.from_pretrained(path_b)

plots = [
    {
        "path_a": "/cs/student/projects3/COMP0087/grp1/models/iter/prune_ft0_87",
        "path_b": "/cs/student/projects3/COMP0087/grp1/models/prune_ft06",
        "name_a": "iter (sparsity: 0.5)",
        "name_b": "prune_finetune (sparsity: 0.5)",
        "file_path": "paper/iter_vs_prft_05"
    },
    {
        "path_a": "/cs/student/projects3/COMP0087/grp1/models/iter/prune_ft0_10",
        "path_b": "/cs/student/projects3/COMP0087/grp1/models/prune_ft01",
        "name_a": "iter (sparsity: 0.08)",
        "name_b": "prune_finetune (sparsity: 0.08)",
        "file_path": "paper/iter_vs_prft_008"
    },
    {
        "path_a": "/cs/student/projects3/COMP0087/grp1/models/ft_prune01",
        "path_b": "/cs/student/projects3/COMP0087/grp1/models/prune_ft01",
        "name_a": "finetune_prune (sparsity: 0.1)",
        "name_b": "prune_finetune (sparsity: 0.08)",
        "file_path": "paper/ftpr_vs_prft_008"
    },
    {
        "path_a": "/cs/student/projects3/COMP0087/grp1/models/ft_prune06",
        "path_b": "/cs/student/projects3/COMP0087/grp1/models/prune_ft06",
        "name_a": "finetune_prune (sparsity: 0.6)",
        "name_b": "prune_finetune (sparsity: 0.5)",
        "file_path": "paper/ftpr_vs_prft_06"
    }
]

for plot in plots:
    print(plot.keys())
    model_a = AutoModelForCausalLM.from_pretrained(plot["path_a"])
    model_b = AutoModelForCausalLM.from_pretrained(plot["path_b"])
    file_name = plot['file_path']
    plot_layers_cumulative_diff(model_a, model_b, plot["name_a"], plot["name_b"], file_name, top_k_percent=5)
    print("Done.")
        
