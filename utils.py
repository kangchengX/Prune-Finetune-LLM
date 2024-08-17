import os
import subprocess
import random
import torch
from datasets import load_dataset
import torch.nn as nn
import transformers
import sys
import evaluate
import numpy as np
from tqdm import tqdm 
import pandas as pd

from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM,AutoTokenizer
from wanda.lib.eval import eval_ppl
import bitsandbytes
import wandb
import json
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftModel


def freeze_weights(model):
    #freezing original weights
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

def find_layers(module, layers=[nn.Linear,bitsandbytes.nn.Linear8bitLt], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res




def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()
        if(sub_params != 0):
            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")
        else:
            print(f"layer {i} sparsity INFTY")

    model.config.use_cache = use_cache 
    return float(count)/total_params 


def get_response(prompt, saved_model:AutoModelForCausalLM, tokenizer:AutoTokenizer, device=torch.device("cuda:0"),max_new_tokens=1):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    inputs = inputs.to(device)
    outputs = saved_model.generate(inputs,
                                    max_new_tokens=max_new_tokens,
                                    do_sample=True,
                                    repetition_penalty=1.18,
                                    temperature=0.01,
                                    top_k=40,
                                    # top_p=0.1,
                                    pad_token_id=tokenizer.eos_token_id
                                    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return response


# parse model response and extract the model'schoice
def parse_choice(response):
    choices=["A","B","C","D"]
    
    if response[-1] in choices:
        return choices.index(response[-1]) + 1
    else:
        return None

def parse_choice_bbh(response):
    choices=["(A)","(B)","(C)","(D)","(E)"]
    
    if response[-1] in choices:
        return choices.index(response[-1]) + 1
    else:
        return None

def finetune(tokenizer, model_name, save_path, i=1, epochs=0.1):

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                torch_dtype=torch.float16,
                                                # low_cpu_mem_usage=True, 
                                                load_in_8bit=True,
                                                device_map='cuda:0')
    tokenizer.pad_token = tokenizer.eos_token
    
    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float16)
    model.lm_head = CastOutputToFloat(model.lm_head)


    #setting up LORA Adapters
    config = LoraConfig(
        r=8, #attention heads (before it was 16)
        # target_modules='all-linear',
        lora_alpha=16, #alpha scaling
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
    )
    model = get_peft_model(model, config)

    #loading dataset
    data = load_dataset("wikitext",'wikitext-2-raw-v1')

    #changing the form of the data
    #def merge_columns(example):
    #    example['text'] = example['text'] + example['text']
    #    return example

    #data['train'] = data['train'].map(merge_columns)
    data = data.map(lambda samples: tokenizer(samples['text'], padding='max_length', truncation= True), batched=True)

    # metric = evaluate.load("accuracy")#evaluate.load("perplexity",module_type="metric")

    # Fine-tune the model
    for _ in range(2):
        trainer_args = TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            num_train_epochs = epochs,
            seed = i,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir=save_path, 
            save_strategy="steps",
            save_steps=200,
            # resume_from_checkpoint=save_path+"/checkpoint-200",
            report_to="wandb" 
        )

        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset= data['train'], # type: ignore
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        model.config.use_cache = False 
        trainer.train(
            # resume_from_checkpoint=True
            )
    
    #We need to first save the adapter model (this is useful if we need to remove the adapter)
    model.save_pretrained(save_path+"/adapter")
    #Then reload it and save it merged:
    merge_peft(tokenizer,save_path)
    

def merge_peft(tokenizer, path):
    with torch.no_grad():
        print("Reload and Merge models")
        peft_model = AutoPeftModelForCausalLM.from_pretrained(path+"/adapter",
                                                torch_dtype=torch.float16,
                                                device_map='cuda:0')
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(path)
        #We also have to store the tokenizer in the merged, study if this is needed or we can just move the 
        tokenizer.save_pretrained(path)
        del peft_model, tokenizer, merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    

def get_llm(model_name, cache_dir="hf_cache", use_8bit=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        load_in_8bit=use_8bit,
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="cuda:0"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def gen_path(path):
    if not os.path.exists(path):
        try:
            # Create the directory
            os.makedirs(path)
            print(f"Directory '{path}' created successfully.")
        except OSError as err:
            print(f"Error creating directory '{path}': {err}")
    else:
        print(f"Directory '{path}' already exists.")

def prune(save_path, model = "baffo32/decapoda-research-llama-7B-hf", sparsity=0.5):
    '''prune the model
    
    Input:
        save_path: path to save the model
        model: hugging face link or local model path
        sparsity: the pruning sparsity
    Output:
        (not return): save the pruned model to save_path
    '''
    # Prune the fine-tuned model
    # Define the command to prune the model using Wanda pruning method
    print("PATH",sys.path)
    prune_command = [
        "python", "wanda/main.py",
        "--model", model,
        "--prune_method", "wanda",
        "--sparsity_ratio", f"{sparsity}",
        "--sparsity_type", "unstructured",
        "--save", "out/llama_7b/unstructured/wanda/",
        "--save_model", save_path
    ]

    # Run the command
    subprocess.run(prune_command)


def initialize_file(file_name:str):
    data = {}
    data["finetune"] = []
    data["prune"] = []
    data["base"] = []
    data["finetune_prune"] = []
    data["prune_finetune"] = []
    data["prune_finetune_iter"] = []
    data["finetune_iter_prune"] = []
    data["iter"] = []
    with open(f'{file_name.split(".")[0]}.json', 'w') as file:
        json.dump(data, file, indent=4)
        

def  write_results_v3(type:str, sparsity_txt, metrics):
    assert type == "finetune" or \
        type == "prune" or \
        type == "base" or \
        type == "finetune_prune" or \
        type == "prune_finetune" or \
        type == "prune_finetune_iter" or \
        type == "finetune_iter_prune" or \
        type == "iter", "Type is not valid"
    if not(os.path.exists("res.json")):
        initialize_file("res.json")

    with open('res.json', 'r') as file:
        data = json.load(file)
        assert type in data, "File is not initialised correctly."
        found = False
        for item in data[type]:
            #We are going to change the value for a specific sparsity
            if item.get("sparsity")==sparsity_txt \
                and item.get("iterations")==metrics["iterations"]:
                print("Found", sparsity_txt, "==", item.get("sparsity"))
                found = True
                item.clear()  # Clear the existing dictionary
                item.update(metrics)
        if not found:
            data[type].append(metrics)
    # Write the updated data back to the file
    with open('res.json', 'w') as file:
        json.dump(data, file, indent=4)

     
def get_last_sparsity_iter():
    with open('res.json', 'r') as file:
        data = json.load(file)
        return data["iter"][-1]["sparsity"]
