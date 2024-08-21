import os, subprocess
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM


class CastOutputToFloat(nn.Sequential):
    """Cast the model outputs to float16."""
    
    def forward(self, x): return super().forward(x).to(torch.float16)


def finetune(
        tokenizer: AutoTokenizer, 
        model: str, 
        save_path: str | None =  None, 
        seed: int | None = 1, 
        epochs: int | float | None = 0.1
):
    """
    Fine-tuning the model using LoRA.

    Args:
        tokenizer (AutoTokenizer): tokenizer for the model.
        model_name (str): Can be either:
            A string with the shortcut name of a pretrained model to load from cache or download, e.g., bert-base-uncased.
            A string with the identifier name of a pretrained model that was user-uploaded to our S3, e.g., dbmdz/bert-base-german-cased.
            A path to a directory containing model weights saved using save_pretrained(), e.g., ./my_model_directory/.
            A path or url to a tensorflow index checkpoint file (e.g, ./tf_model/model.ckpt.index). In this case, from_tf should be set to True and a configuration object should be provided as config argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
        seed (int): seed during training.
        save_path (str): path to save the model.
        epochs (float): number of epochs.  
    """
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map='cuda:0'
    )
    # replace pad token with eos (end of sequence token)
    tokenizer.pad_token = tokenizer.eos_token
    
    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads() # enable gradients with respect to the embedding
    model.lm_head = CastOutputToFloat(model.lm_head)

    # setting up LORA Adapters
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)

    #loading dataset
    data = load_dataset("wikitext",'wikitext-2-raw-v1')
    data = data.map(lambda samples: tokenizer(samples['text'], padding='max_length', truncation= True), batched=True)

    # Fine-tune the model
    trainer_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs = epochs,
        seed=seed,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir=save_path, 
        save_strategy="steps",
        save_steps=200
    )

    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset= data['train'], # type: ignore
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False # don't store key and value states 
    trainer.train()
    
    if save_path is not None:
        # Save the adapter model (this is useful if we need to remove the adapter)
        model.save_pretrained(save_path+"/adapter")
        #Then reload it and save it merged:
        merge_peft(tokenizer, save_path)


def prune(model: str | None = "baffo32/decapoda-research-llama-7B-hf", save_path: str | None = None, sparsity: float | None = 0.5):
    """
    Prune the model.
    
    Args:
        model (str): Can be either:
            A string with the shortcut name of a pretrained model to load from cache or download, e.g., bert-base-uncased.
            A string with the identifier name of a pretrained model that was user-uploaded to our S3, e.g., dbmdz/bert-base-german-cased.
            A path to a directory containing model weights saved using save_pretrained(), e.g., ./my_model_directory/.
            A path or url to a tensorflow index checkpoint file (e.g, ./tf_model/model.ckpt.index). In this case, from_tf should be set to True and a configuration object should be provided as config argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
            Default to `"baffo32/decapoda-research-llama-7B-hf"`.

        save_path (str): path to save the model.
        sparsity (float): the pruning sparsity.
    """

    # Prune the fine-tuned model
    # Define the command to prune the model using Wanda pruning method
    prune_command = [
        "python", os.path.join(os.path.dirname(__file__), "wanda/main.py"),
        "--model", model,
        "--prune_method", "wanda",
        "--sparsity_ratio", str(sparsity),
        "--sparsity_type", "unstructured"
    ]

    if save_path is not None:
        prune_command += ["--save_model", save_path]

    # Run the command
    subprocess.run(prune_command)


def merge_peft(tokenizer: AutoTokenizer, path: str):
    """
    Merge the model from peft model.
    
    Args:
        tokenizer (AutoTokenizer): tokenizer.
        path (str): path to save the model and the peft model is in path/adapter.
    """
    with torch.no_grad():
        print("Reload and Merge models")
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
            path+"/adapter",
            torch_dtype=torch.float16,
            device_map='cuda:0'
        )
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(path)
        #We also have to store the tokenizer in the merged, study if this is needed or we can just move the 
        tokenizer.save_pretrained(path)
        del peft_model, tokenizer, merged_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
