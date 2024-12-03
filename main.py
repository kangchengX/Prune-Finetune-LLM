import os, argparse, warnings
from transformers import AutoTokenizer
from wanda.lib.prune import check_sparsity
from utils import get_llm, write_results
from eval import eval_model
from process import finetune, prune


cache_path = './hf_cache/'
os.environ['HF_HOME']=cache_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparsity', type=float, default=0.0, help='sparsity')
    parser.add_argument('--action', type=str, default="base", help='action to perform')
    parser.add_argument('--out_type', type=str, default="base", help='Type of the output in write.txt')
    parser.add_argument('--model_path', type=str, default="baffo32/decapoda-research-llama-7B-hf", help='store path')
    parser.add_argument('--save_path', type=str, default="baffo32/decapoda-research-llama-7B-hf", help='save path')
    parser.add_argument('--auto_tokenizer_model_name', type=str, default="baffo32/decapoda-research-llama-7B-hf", help='name or path of the model from which the tokenizer will be inferred by autotokenizer. ')
    parser.add_argument('--eval', type=str, default="True", help='evaluate model')
    parser.add_argument('--epochs', type=float, default=0.1, help='finetuning epochs')
    parser.add_argument('--ft_iter', type=int, default=1, help='ith iteration for this finetuning if action = finetune, number of times fine-tuning has been performed if action = prune.')
    parser.add_argument('--results_path', type=str, default='results.json', help='path to save the results as a json file')

    args = parser.parse_args()

    args.eval = eval(args.eval)
    
    tokenizer = AutoTokenizer.from_pretrained(args.auto_tokenizer_model_name, use_fast = False)

    os.makedirs(args.save_path, exist_ok=True)
    
    if args.action == "finetune":
        finetune(tokenizer=tokenizer, model=args.model_path, save_path=args.save_path, seed=args.ft_iter, epochs=args.epochs)
    elif args.action == "prune":
        prune(model=args.model_path, save_path=args.save_path, sparsity=args.sparsity)
    elif args.action == "base":
        if args.model_path != args.save_path:
            warnings.warn('--model_path and --save_path are not the same, which may casue unexpected behaviour, since model evaluation is based on --save_path.')
    else:
        raise ValueError('Unsupported --action : {}'.format(args.action))
    
    
    metrics = {"finetune_iterations": args.ft_iter}

    if args.eval:
        saved_model = get_llm(args.save_path)
        sparsity_latest = check_sparsity(saved_model) # check the sparsity of the final model and compare sparsities
        accuracy_mmlu = eval_model(saved_model, tokenizer, ds_name="cais/mmlu")
        accuracy_bbh = eval_model(saved_model, tokenizer, ds_name="lukaemon/bbh")
        accuracy_belebele = eval_model(saved_model, tokenizer, ds_name="facebook/belebele")
        accuracy_factoid_qa = eval_model(saved_model, tokenizer, ds_name="kelvin-jiang/factoid-qa")
        ppl = eval_model(saved_model, tokenizer, ds_name='wikitext2')
        metrics["ppl"] = ppl
        metrics["bbh"] = accuracy_bbh
        metrics["mmlu"] = accuracy_mmlu
        metrics["belebele"] = accuracy_belebele
        metrics["factoid_qa"] = accuracy_factoid_qa
    
    metrics["sparsity_prune"] = round(args.sparsity, 2)
    metrics["sparsity_latest"] = round(sparsity_latest, 2)

    write_results(pipeline=args.out_type, metrics=metrics, results_path=args.results_path)
