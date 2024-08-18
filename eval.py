import torch, re, os, warnings
from datasets import load_dataset
from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Literal, List
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from factoid_qa.freebase_qa import FreebaseQA
from utils import get_response,parse_choice


def eval_model(
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        ds_name: Literal["facebook/belebele", "lukaemon/bbh", "cais/mmlu", 'kelvin-jiang/factoid-qa'],
        device: torch.device | None = torch.device("cuda:0"), 
        num_prompts: int | None = None, 
        qa_data_path: str | None = os.path.join(os.path.dirname(__file__), "factoid_qa/FreebaseQA-eval.json")
):
    """
    Evaluate model's performance by metric determined by `ds_name`.

    Args:
        model (AutoModelForCausalLM): the model to evaluate.
        tokenizer (AutoTokenizer): tokenizer.
        ds_name (str): name of the dataset.
        device (device): device to load dataset to.
        num_prompts (int): number of prompts to feed to the model.
        qa_data_path (str): path of the data for factoid qa.

    Returns:
        out (float): value of the metric.
    """
    model.eval()
    if ds_name == "facebook/belebele":
        return eval_belebele(model, tokenizer, device=device, num_prompts=num_prompts)
    elif ds_name == "lukaemon/bbh":
        return eval_bbh_logical_deduction_five(model, tokenizer, device=device, num_prompts=num_prompts)
    elif ds_name == "cais/mmlu":
        return eval_mmlu(model, tokenizer, device=device, num_prompts=num_prompts)
    elif ds_name == 'kelvin-jiang/factoid-qa':
        if num_prompts is None:
            num_prompts == 600
        return qa_accuracy(model, tokenizer, device=device, num_prompts=num_prompts, freebase_filepath=qa_data_path)
    else:
        raise ValueError('Unsupported ds_name {}'.format(ds_name))
    

def select_dataset(
        ds: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
        num_examples: int | None = 5,
        num_prompts: int | None = None
):
    """
    Select examples and prompts for inference.

    Args:
        ds (DatasetDict | Dataset | IterableDatasetDict |IterableDataset): the loaded dataset.
        num_examples (int): number of examples.
        num_prompts (int): number of prompts for inference.

    Returns:
        ds_examples (Dataset): the examples.
        ds_prompts (Dataset): the prompts.
    """
    ds_length = len(ds)
    if num_examples > ds_length // 2:
        warnings.warn('Examples are more than half of the dataset')
    if num_prompts is None:
        num_prompts = ds_length - num_examples
    if num_examples + num_prompts > ds_length:
        raise ValueError('Total examples exceed ds length')
    
    ds_examples=ds.select(range(0, num_examples))
    ds_prompts=ds.select(range(num_examples, num_prompts + num_examples))

    return ds_examples, ds_prompts


def eval_mmlu(
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        device: torch.device | None = torch.device("cuda:0"), 
        zero_shot: bool | None = False,
        num_prompts: int | None = None
):
    """
    Evaluate mmlu.
    """
    ds = load_dataset("cais/mmlu", split="test", name="all")
    ds_examples, ds_prompts=select_dataset(ds, num_examples=5, num_prompts=num_prompts)

    prompt_template = """
    Question: {question}
    Answer A: {choices[0]}
    Answer B: {choices[1]}
    Answer C: {choices[2]}
    Answer D: {choices[3]}
    Correct answer: {target}"""

    # prepare example prompts
    choices=["A","B","C","D"]
    prompt_examples = ""
    if not zero_shot:
        prompt_examples = "\n\n".join([prompt_template.format(**d,target=choices[int(d["answer"])]) for d in ds_examples])
    
    # Loop through prompts and evaluate model responses
    q_correct = q_total = 0
    for rowNo, row in enumerate(tqdm(ds_prompts)):        
        # Construct the prompt by combining the example prompts and the current row's question
        prompt = (prompt_examples + "\n\n" + prompt_template.format(**row, target="")).strip()
        max_new_tokens=1
        response = get_response(prompt, model, tokenizer, device, max_new_tokens)

        match = re.findall(r'Correct answer: \(?[A-D]', response[0])
        if len(match)>=1:
            choice = match[-1][-1]
        else:
            choice = "(NO CHOICE)"

        # Parse the model's choice and compare it to the correct answer
        if choice == choices[int(row["answer"])]:
            q_correct+=1 
        q_total+=1

    accuracy = q_correct/q_total*100
    return accuracy


def eval_bbh_logical_deduction_five(
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        device: torch.device | None = torch.device("cuda:0"), 
        zero_shot: bool | None = True,
        num_prompts: int | None = None
):
    """
    Evaluate bbh.
    """
    ds = load_dataset("lukaemon/bbh", "logical_deduction_five_objects", split="test")
    ds_examples, ds_prompts=select_dataset(ds, num_examples=5, num_prompts=num_prompts)

    prompt_template = """
    Question: {input}
    Correct answer: {target}"""

    # prepare example prompts
    choices = ["(A)","(B)","(C)","(D)","(E)"]
    prompt_examples = ""
    if not zero_shot:
        prompt_examples = "\n\n".join([prompt_template.format(**d,correct_answer=choices[int(d["correct_answer_num"])-1]) for d in ds_examples])
    tokenizer(prompt_examples, return_tensors="pt")

    # Loop through prompts and evaluate model responses
    q_correct = q_total = 0
    for rowNo, row in enumerate(tqdm(ds_prompts)):        
        # Construct the prompt by combining the example prompts and the current row's question
        prompt = (prompt_examples + "\n\n" + prompt_template.format(input=row["input"], target="")).strip()
        max_new_tokens=3
        response = get_response(prompt, model, tokenizer, device, max_new_tokens)

        # Generate a response from the model
        match = re.findall(r'Correct answer: \(?[A-E]', response[0])
        if len(match)>=1:
            choice = match[-1][-1]
        else:
            choice = "(NO CHOICE)"

        # Parse the model's choice and compare it to the correct answer
        # choice = parse_choice_bbh(response[-3:].strip())
        if choice == row["target"][1]:
            q_correct+=1 
        q_total+=1

    accuracy = q_correct/q_total*100
    return accuracy


def eval_belebele(
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        device: torch.device | None = torch.device("cuda:0"), 
        num_prompts: int | None = None
):
    """
    Evalute belebele.
    """
    ds = load_dataset(path="facebook/belebele", name="eng_Latn", split="test")
    ds_examples, ds_prompts=select_dataset(ds, num_examples=5, num_prompts=num_prompts)

    prompt_template="""{flores_passage}
    Question: {question}
    Answer A: {mc_answer1}
    Answer B: {mc_answer2}
    Answer C: {mc_answer3}
    Answer D: {mc_answer4}
    Correct answer: {correct_answer}"""

    # Prepare example prompts for 5-shot prompting
    choices=["A","B","C","D"]
    prompt_examples = "\n\n".join([prompt_template.format(**d,correct_answer=choices[int(d["correct_answer_num"])-1]) for d in ds_examples])

    # print(prompt_examples)
    tokenizer(prompt_examples, return_tensors="pt")

    # Loop through prompts and evaluate model responses
    q_correct = q_total = 0
    for rowNo, row in enumerate(tqdm(ds_prompts)):        
        # Construct the prompt by combining the example prompts and the current row's question
        prompt_examples = "\n\n".join([prompt_template.format(**d,correct_answer=choices[int(d["correct_answer_num"])-1]) for d in ds_examples])
        prompt = (prompt_examples + "\n\n" + prompt_template.format(**row, correct_answer="")).strip()

        response = get_response(prompt, model, tokenizer, device)
        match = re.findall(r'Correct answer: [A-D]', response[0])
        if len(match)>=6:
            choice = match[-1]
        else:
            choice = "(NO CHOICE)"

        # Parse the model's choice and compare it to the correct answer
        choice=parse_choice(choice.strip())#response[0].strip())
        if choice==int(row["correct_answer_num"]):
            q_correct+=1 
        q_total+=1

    # print(f"{q_total} questions, {q_correct} correct ({round(q_correct/q_total*100,1)}%)")  
    accuracy = q_correct/q_total*100
    return accuracy


def validate_response(correct_answer: List[str], generated_output: str):
    generated_output = generated_output.strip().replace(" ", "").lower()
    correct_answer = [item.strip().replace(" ", "").lower() for item in correct_answer]
    for ans in correct_answer:
        if ans in generated_output: return 1
    return 0


def qa_accuracy(
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        device: torch.device | None = torch.device("cuda:0"), 
        freebase_filepath: str | None = os.path.join(os.path.dirname(__file__), "factoid_qa/FreebaseQA-eval.json"),
        num_prompts: int | None = 600
):
    """
    Evaluate model's performance by factoid qa.
    """
    freebase_qa = FreebaseQA()
    model.eval()
                
    exact_match = 0
    num_prompts_fed = 0

    for question, answers in freebase_qa._generate_examples(freebase_filepath):
        if num_prompts_fed > num_prompts: 
            break
        
        lamma_prompt = f"Please give answer to this question: {question}\nThe answer is "
        inputs = tokenizer(lamma_prompt, return_tensors="pt")
        inputs = inputs.to(device)

        generate_ids = model.generate(inputs.input_ids, max_length=inputs["input_ids"].shape[-1] * 3)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        is_match = validate_response(answers, output)
        exact_match += is_match
        num_prompts_fed += 1

    return(exact_match/num_prompts)

