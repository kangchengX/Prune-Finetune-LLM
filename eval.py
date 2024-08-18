import torch, re
from datasets import load_dataset
from tqdm import tqdm 
from transformers import AutoTokenizer
from .factoid_qa.freebase_qa import FreebaseQA
from .utils import get_response,parse_choice


def eval_model(saved_model, tokenizer, device, ds_name=None, qa_num_examples=600, qa_data_path="FreebaseQA-train.json"):
    if ds_name == "facebook/belebele":
        return eval_belebele(saved_model, tokenizer, device=device)
    elif ds_name == "lukaemon/bbh":
        return eval_bbh_logical_deduction_five(saved_model, tokenizer, device=device)
    elif ds_name == "cais/mmlu":
        return eval_mmlu(saved_model, tokenizer, device=device)
    elif ds_name == 'kelvin-jiang/factoid-qa':
        return qa_accuracy(saved_model, tokenizer, device, num_examples=qa_num_examples, freebase_filepath=qa_data_path)
    else:
        return None
    

def eval_mmlu(saved_model, tokenizer:AutoTokenizer, device=torch.device("cuda:0"), zero_shot=False):
    ds = load_dataset("cais/mmlu", split="test", name="all")
    
    ds_examples=ds.select(range(0,5))
    ds_prompts=ds.select(range(5,len(ds)))

    prompt_template = """
    Question: {question}
    Answer A: {choices[0]}
    Answer B: {choices[1]}
    Answer C: {choices[2]}
    Answer D: {choices[3]}
    Correct answer: {target}"""

    #prepare example prompts
    choices=["A","B","C","D"]
    prompt_examples = ""
    if not zero_shot:
        prompt_examples = "\n\n".join([ prompt_template.format(**d,target=choices[int(d["answer"])]) for d in ds_examples])
    
    # Loop through prompts and evaluate model responses
    q_correct = q_total = 0
    for rowNo, row in enumerate(tqdm(ds_prompts)):        
        # Construct the prompt by combining the example prompts and the current row's question
        prompt = (prompt_examples + "\n\n" + prompt_template.format(**row, target="")).strip()
        max_new_tokens=1
        response = get_response(prompt, saved_model, tokenizer, device, max_new_tokens)

        # Generate a response from the model
        # match = re.findall(r'\([A-E]\)', response[0])
        match = re.findall(r'Correct answer: \(?[A-D]', response[0])
        if len(match)>=1:
            choice = match[-1][-1]
        else:
            choice = "(NO CHOICE)"

        # Parse the model's choice and compare it to the correct answer
        # choice = parse_choice_bbh(response[-3:].strip())
        if choice == choices[int(row["answer"])]:
            q_correct+=1 
        q_total+=1

    accuracy = q_correct/q_total*100
    return accuracy

def eval_bbh_logical_deduction_five(saved_model, tokenizer:AutoTokenizer, device=torch.device("cuda:0"), zero_shot=True):
    ds = load_dataset("lukaemon/bbh", "logical_deduction_five_objects",split="test")
    
    ds_examples=ds.select(range(0,5))
    ds_prompts=ds.select(range(5,len(ds)))

    prompt_template = """
    Question: {input}
    Correct answer: {target}"""

    #prepare example prompts
    choices = ["(A)","(B)","(C)","(D)","(E)"]
    prompt_examples = ""
    if not zero_shot:
        prompt_examples = "\n\n".join([ prompt_template.format(**d,correct_answer=choices[int(d["correct_answer_num"])-1]) for d in ds_examples])
    tokenizer(prompt_examples, return_tensors="pt")

    # Loop through prompts and evaluate model responses
    q_correct = q_total = 0
    for rowNo, row in enumerate(tqdm(ds_prompts)):        
        # Construct the prompt by combining the example prompts and the current row's question
        prompt = (prompt_examples + "\n\n" + prompt_template.format(input=row["input"], target="")).strip()
        max_new_tokens=3
        response = get_response(prompt, saved_model, tokenizer, device, max_new_tokens)

        # Generate a response from the model
        # match = re.findall(r'\([A-E]\)', response[0])
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

def parse_choice_bbh(response):
    return response[-3:]

def eval_belebele(saved_model, tokenizer:AutoTokenizer, device=torch.device("cuda:0")):
    ds = load_dataset(path="facebook/belebele", name="eng_Latn", split="test")

    ds_examples = ds.select(range(0,5))
    ds_prompts = ds.select(range(5, len(ds)))

    prompt_template="""{flores_passage}
    Question: {question}
    Answer A: {mc_answer1}
    Answer B: {mc_answer2}
    Answer C: {mc_answer3}
    Answer D: {mc_answer4}
    Correct answer: {correct_answer}"""

    # Prepare example prompts for 5-shot prompting
    choices=["A","B","C","D"]
    prompt_examples = "\n\n".join([ prompt_template.format(**d,correct_answer=choices[int(d["correct_answer_num"])-1]) for d in ds_examples])

    # print(prompt_examples)
    tokenizer(prompt_examples, return_tensors="pt")

    # Loop through prompts and evaluate model responses
    q_correct = q_total = 0
    for rowNo, row in enumerate(tqdm(ds_prompts)):        
        # Construct the prompt by combining the example prompts and the current row's question
        
        ds_examples = ds.shuffle(seed=rowNo).select(range(0,5))
        prompt_examples = "\n\n".join([prompt_template.format(**d,correct_answer=choices[int(d["correct_answer_num"])-1]) for d in ds_examples])
        prompt = (prompt_examples + "\n\n" + prompt_template.format(**row, correct_answer="")).strip()

        response = get_response(prompt, saved_model, tokenizer, device)
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


def validate_response(correct_answer, generated_output):
    generated_output = generated_output.strip().replace(" ", "").lower()
    correct_answer = [item.strip().replace(" ", "").lower() for item in correct_answer]
    for ans in correct_answer:
        if ans in generated_output: return 1
    return 0


def qa_accuracy(model, tokenizer, device, freebase_filepath = "FreebaseQA-train.json", num_examples=10):

    freebase_qa = FreebaseQA()
    model.eval()
                
    exact_match = 0
    num_sample = 0

    for question, answers in freebase_qa._generate_examples(freebase_filepath):
        if num_sample > num_examples: break
        
        lamma_prompt = f"Please give answer to this question: {question}\nThe answer is "
        inputs = tokenizer(lamma_prompt, return_tensors="pt")
        inputs = inputs.to(device)

        generate_ids = model.generate(inputs.input_ids, max_length=inputs["input_ids"].shape[-1] * 3)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        is_match = validate_response(answers, output)
        exact_match += is_match
        num_sample += 1
                
    print(f"Exact match: {exact_match}/{num_examples} || Accuracy : {100 * (exact_match/num_examples):.2f}%")

    return(exact_match/num_examples)
