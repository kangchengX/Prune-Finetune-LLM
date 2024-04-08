import os
import pandas as pd
cache_path = './hf_cache/'

os.environ['HF_HOME']=cache_path
os.environ['TRANSFORMERS_CACHE'] = cache_path

import torch
import matplotlib.pyplot as plt
import numpy as np

from utils import get_llm, find_layers
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM


def find_weights(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    if isinstance(model, PeftModelForCausalLM):
        layers = model.model.model.layers
    else:
        layers = model.model.layers

    keys = ['self_attn.q_proj.lora_A.default', 'self_attn.q_proj.lora_B.default', 'self_attn.v_proj.lora_A.default', 'self_attn.v_proj.lora_B.default']
    weights = []

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        subset_lora = {
            key: value
            for key, value in subset.items()
            if 'lora' in key
        }
        assert len(subset_lora) == 4

        weight_adapter_q = subset_lora[keys[1]].weight.data @ subset_lora[keys[0]].weight.data
        weight_adapter_v = subset_lora[keys[3]].weight.data @ subset_lora[keys[2]].weight.data

        weights.append((weight_adapter_q,weight_adapter_v))


def cal_metrics(weights):
    metrics = ([],[])
    for weights_qv in weights:
        weights_q = weights_qv[0]
        weights_v = weights_qv[1]
        dict_q = {'mean':weights_q.mean().item(),
                  'std':weights_q.std().item(),
                  'max':weights_q.max().item(),
                  'min':weights_q.min.item()}
        dict_v = {'mean':weights_v.mean().item(),
                  'std':weights_v.std().item(),
                  'max':weights_v.max().item(),
                  'min':weights_v.min.item()}
        metrics[0].append(dict_q)
        metrics[1].append(dict_v)

    return metrics


def plot_iteration(metrics_iter,styles):
    assert len(metrics_iter) == len(styles)
    fig, axs = plt.subplots(4, 2, figsize=(12, 24))
    # for each iter
    for i,metrics in enumerate(metrics_iter):
        # for each layer
        for j,metrics_layer in enumerate(metrics):
            mean = [metric['mean'] for metric in metrics_layer]
            std = [metric['std'] for metric in metrics_layer]
            max = [metric['max'] for metric in metrics_layer]
            min = [metric['min'] for metric in metrics_layer]
            metrics_draws = [mean,std,max,min]
            # for each kind of metric
            for k,metrics_draw in enumerate(metrics_draws):
                axs[k][j].plot(range(1,1+len(metrics_draw)),metrics_draw,
                               styles[i],markersize=5,markerfacecolor='none', label = 'iter' + str(i))
                axs[k][j].set_xlabel('layers')
                axs[k][j].set_ylabel('metric value')
                axs[k][j].legend()

    axs[0][0].set_title('mean_q');axs[0][1].set_title('mean_v')
    axs[1][0].set_title('std_q');axs[1][1].set_title('std_v')
    axs[2][0].set_title('max_q');axs[2][1].set_title('max_v')
    axs[3][0].set_title('min_q');axs[3][1].set_title('min_v')

    fig.suptitle('for model')

    plt.savefig('test.png')

def cal_metrics_iter(model_paths:list):
    metrics_iter = []
    for model_path in model_paths:
        model = get_llm(model_path)
        weights = find_weights(model)
        metrics = cal_metrics(weights)
        metrics_iter.append(metrics)
        


get_llm("baffo32/decapoda-research-llama-7B-hf")