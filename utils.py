import os, json, math, warnings
import torch, bitsandbytes
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Type, List


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


def parse_choice(response: str, choices: List[str] | None = ["A","B","C","D"]):
    """
    Takes a response string and returns the index of the choices or None if the choice is not valid.
    
    Args:
        response (str): LLM's response.
        choices (list): list of choices to select from.
    
    Returns:
        out (int | None): Index of the choice selected, starting from 1, if the last character of the response is in `choices`.\
            If the last character is not one of these choices, returns `None`.
    """
    
    if response[-1] in choices:
        return choices.index(response[-1]) + 1
    else:
        return None
    

def get_llm(model_name: str, use_8bit : bool | None = False):
    """
    Load LLM from `model_name` to device. Assign `model.config.max_position_embeddings` to `model.seqlen`.

    Args:
        model_name (str): path or name of the model.
        use_8bit (bool): if to load model in 8bit.

    Returns:
        model (AutoModelForCausalLM): the model.

    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        load_in_8bit=use_8bit,
        low_cpu_mem_usage=True, 
        device_map="cuda:0"
    )

    model.seqlen = model.config.max_position_embeddings 

    return model


def validate_response(correct_answer: List[str], generated_output: str):
    """
    Determine if any correct answer is present in the generated output.
    
    Args:
        correct_answer (list): list of strings that represent the expected correct answers.
        generated_output (str): response for LLM.
    
    Returns:
        out (bool): `True` if any correct answer is found in the generated output; `False` otherwise.
    """
    generated_output = generated_output.strip().replace(" ", "").lower()
    correct_answer = [item.strip().replace(" ", "").lower() for item in correct_answer]
    for ans in correct_answer:
        if ans in generated_output: return True
    return False


def write_results(pipeline: str, metrics: dict, results_path: str | None = "results.json"):
    """
    Write results to the file.

    Args:
        pipeline (str): name of the pipeline.
        metrics (dict): dict with keys `"ppl"`, `"bbh"`, `"mmlu"`, `"belebele"`, `"factoid_qa"`, \
            `"sparsity_prune"`, `"sparsity_latest"` and `"ft_iter"`.
        results_path (str): path to save the results as a json file.
    """

    results_path = os.path.splitext(results_path)[0] + '.json'
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            data = json.load(f)
    else:
        data = dict()

    append_results = True

    # data has type Dict[str(pipeline), List[Dict[str(metric), int | float]]]
    for record in data.setdefault(pipeline, []):
        # change the value for a specific sparsity and ft_iter
        if math.isclose(record["sparsity_prune"], metrics["sparsity_prune"]) and record["finetune_iterations"] == metrics["finetune_iterations"]:
            warnings.warn(f"Found pipeline : {pipeline}, sparsity_prune : {metrics['sparsity_prune']}, \
                          finetune_iterations : {metrics['finetune_iterations']}. The results will be overwritten.")
            record = metrics
            append_results = False
            break
    
    if append_results:
        data[pipeline].append(metrics)

    # Write the updated data back to the file
    with open(results_path, 'w') as f:
        json.dump(data, f, indent=4)
