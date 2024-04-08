import os
import argparse
import sys

import wandb
cache_path = './hf_cache/'

os.environ['HF_HOME']=cache_path
os.environ['TRANSFORMERS_CACHE'] = cache_path

import torch
from transformers import AutoTokenizer
from wanda.lib.eval import eval_ppl, eval_zero_shot
from wanda.lib.prune_opt import check_sparsity
from utils import finetune, gen_path, prune, get_llm, write_results,write_results_v3,plot_outliers
from eval import eval_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparsity', type=str, help='sparsity', default=0.0)
    parser.add_argument('--train', type=str, help='whether to prune/finetune or not', default="True")
    parser.add_argument('--action', type=str, help='action to perform', default="base")
    parser.add_argument('--out_type', type=str, help='Type of the output in write.txt', default="base")
    parser.add_argument('--model_path', type=str, help='store path', default="baffo32/decapoda-research-llama-7B-hf")
    parser.add_argument('--save_path', type=str, help='save path', default="baffo32/decapoda-research-llama-7B-hf")
    parser.add_argument('--eval_ppl', type=str, help='evaluate perplexity', default="True")
    parser.add_argument('--check_sparsity', type=str, help='check sparsity', default="True")
    parser.add_argument('--epochs', type=float, help='finetuning epochs', default=0.1)
    parser.add_argument('--i', type=int, help='Number of iterations to run finetune', default=1)
    args = parser.parse_args()
    args.eval_ppl = eval(args.eval_ppl)
    args.check_sparsity = eval(args.check_sparsity)
    sparsity_txt = "0" + str(args.sparsity)[2:]
    tokenizer = AutoTokenizer.from_pretrained("baffo32/decapoda-research-llama-7B-hf", use_fast = False)

    saved_model = get_llm("./models/ft_prune05")
    sparsity = plot_outliers(saved_model)