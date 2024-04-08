import os
import argparse
import sys
from factoid_qa import qa_accuracy

import wandb
cache_path = './hf_cache/'

os.environ['HF_HOME']=cache_path
os.environ['TRANSFORMERS_CACHE'] = cache_path

import torch
from transformers import AutoTokenizer
from wanda.lib.eval import eval_ppl, eval_zero_shot
from wanda.lib.prune_opt import check_sparsity
from utils import finetune, gen_path, prune, get_llm, write_results_v3
from eval import eval_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparsity', type=float, help='sparsity', default=0.0)
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
    args.train = eval(args.train)
    sparsity_txt = "0" + str(args.sparsity)[2:]
    tokenizer = AutoTokenizer.from_pretrained("baffo32/decapoda-research-llama-7B-hf", use_fast = False)

    if args.action == "finetune" and args.train:
        finetune(tokenizer, args.model_path, args.save_path, args.i, args.epochs)
    elif args.action == "prune" and args.train:
        prune(args.save_path, model=args.model_path, sparsity=args.sparsity)
    
    saved_model = None
    metrics = {"iterations": round(args.i*args.epochs,2)}
    sparsity = args.sparsity
    if args.check_sparsity:
        if saved_model == None:
            saved_model = get_llm(args.save_path)
        sparsity = check_sparsity(saved_model)
        if(args.out_type=="prune_finetune" or args.out_type=="prune_finetune_iter"):
            metrics["added_sparsity"] = args.sparsity - sparsity
        print(sparsity)
    if args.eval_ppl:
        if saved_model == None:
            saved_model = get_llm(args.save_path)
        accuracy_mmlu = eval_model(saved_model, tokenizer, torch.device("cuda:0"), ds_name="cais/mmlu")
        accuracy_bbh = eval_model(saved_model, tokenizer, torch.device("cuda:0"), ds_name="lukaemon/bbh")
        accuracy_belebele = eval_model(saved_model, tokenizer, torch.device("cuda:0"), ds_name="facebook/belebele")
        accuracy_factoid_qa = qa_accuracy(saved_model,tokenizer,torch.device("cuda:0"),num_examples=600)
        ppl = eval_ppl(args, saved_model, tokenizer, device=torch.device("cuda:0"))
        metrics["ppl"] = ppl
        metrics["bbh"] = accuracy_bbh
        metrics["mmlu"] = accuracy_mmlu
        metrics["belebele"] = accuracy_belebele
        metrics["factoid_qa"] = accuracy_factoid_qa
        if args.out_type == "iter":
            metrics["added_sparsity"] = 0.1
    
    metrics["sparsity"] = round(sparsity,2)
    write_results_v3(args.out_type, args.sparsity, metrics)

    # wandb.log({"ppl":ppl_test})

    print("Doing chmod")
    # os.system("chmod -Rf 770 ./")