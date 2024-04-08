import os 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from freebase_qa import FreebaseQA

cache_path = './hf_cache/'

os.environ['HF_HOME']=cache_path
os.environ['TRANSFORMERS_CACHE'] = cache_path

def validate_response(correct_answer, generated_output):
    generated_output = generated_output.strip().replace(" ", "").lower()
    correct_answer = [item.strip().replace(" ", "").lower() for item in correct_answer]
    for ans in correct_answer:
        if ans in generated_output: return 1
    return 0

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = 2048
    return model

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


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("baffo32/decapoda-research-llama-7B-hf", use_fast = False)
    saved_model = get_llm("baffo32/decapoda-research-llama-7B-hf")

    result = qa_accuracy(model=saved_model, tokenizer=tokenizer,device=torch.device("cuda:0"), freebase_filepath = "FreebaseQA-train.json", num_examples=10)
