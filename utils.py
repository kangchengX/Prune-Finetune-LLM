import os, json
import torch, bitsandbytes
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from typing import Type, List


def freeze_weights(model: nn.Module):
    #freezing original weights
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)


def find_layers(module: nn.Module, layers: list[Type] | None = [nn.Linear,bitsandbytes.nn.Linear8bitLt], name: str | None = ''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (Module): PyTorch module.
        layers (list): List of layer types to find. Default to `[nn.Linear,bitsandbytes.nn.Linear8bitLt]`.
        name (str): Name of the module. Default to `''`.

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


def get_response(
        prompt: str, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        device: torch.device | None = torch.device("cuda:0"), 
        max_new_tokens: int | None = 1
) -> str:
    """
    Generates a response from a given prompt durning model inference.

    Args:
        prompt (str): The input text prompt to generate a response for.
        model (AutoModelForCausalLM): The pre-trained language model to use for generation.
        tokenizer (AutoTokenizer): The tokenizer associated with the model for processing the input prompt.
        device (device): The device to run the model on. Defaults to `torch.device("cuda:0")`.
        max_new_tokens (int): The maximum number of new tokens to generate. Defaults to `1`.

    Returns:
        str: The generated response text.
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.seqlen).input_ids
    inputs = inputs.to(device)
    outputs = model.generate(
        inputs,
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


def validate_response(correct_answer: List[str], generated_output: str):
    generated_output = generated_output.strip().replace(" ", "").lower()
    correct_answer = [item.strip().replace(" ", "").lower() for item in correct_answer]
    for ans in correct_answer:
        if ans in generated_output: return 1
    return 0


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
        

def write_results(type:str, sparsity_txt, metrics):
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
