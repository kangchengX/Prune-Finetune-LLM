import subprocess, os, argparse, random
import numpy as np
import torch
from constant import PIPELINE_SAVE_PATH_MAP, PYTHON_INTER

torch.manual_seed(87)
np.random.seed(87)
random.seed(87)

cache_path = './hf_cache/'
os.environ['HF_HOME'] = cache_path
os.environ['TRANSFORMERS_CACHE'] = cache_path


def check_use_saved_model(use_saved_model: bool | str, saved_model_path: str):
    """
    Determine if to use the saved model, only `use_saved_model` is `True` and the model path (or name in the hugging face) exists.
    Be careful when using the saved model. Due to the limitation of storage, a model folder for pipline with `finetune_iter` 
    will be overwritten for each fine-tuning iteration.
    
    Args:
        use_saved_model (bool): if to use saved model if possible.
        saved_model_path (str): path of the model folder.

    Returns:
        out (bool): if to use the saved model.
    """

    if isinstance(use_saved_model, str):
        use_saved_model = eval(use_saved_model)
    return use_saved_model and os.path.exists(saved_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pipeline', 
        type=str,
        default="base",
        help="""Pipeline to run, can be:
        "base": do not prune or fine-tune the model.
        "finetune": only finetune the model for one time.
        "finetune_prune": fine-tune -> prune.
        "finetune_iter": fine-tune x L.
        "finetune_iter_prune": (fine-tune) x L -> prune.
        "prune": only prune the model for one time.
        "prune_finetune": prune -> fine-tune.
        "prune_finetune_iter": prune -> (fine-tune) x L.
        "iter_pf": (prune -> fine-tune) x L.
        "iter_fp": (fine-tune -> prune) x L.
        """
    )
    parser.add_argument(
        '--ft_iter', 
        type=int, 
        default=5, 
        help='Number of iterations to run finetune in the selected pipeline.'
    )
    parser.add_argument(
        '--use_saved_model', 
        type=str, 
        default='True', 
        help="""If True, some fine-tuning or pruning in one pipeline will be executed based on
        the saved models from the previous executed pipelines.
        """
    )
    parser.add_argument(
        '--pretrained_model', 
        type=str, 
        default="baffo32/decapoda-research-llama-7B-hf",
        help="Name of the pretrained model in the hugging face."
    )

    args = parser.parse_args()
    args.use_saved_model = eval(args.use_saved_model)

    current_file_directory = os.path.dirname(__file__)
    main_file_full_path = os.path.join(current_file_directory, 'main.py')
    save_paths = {pipeline: os.path.join(current_file_directory, path) for pipeline, path in PIPELINE_SAVE_PATH_MAP.items()}

    if args.pipeline == 'base':
        # do not prune or fine-tune the model
        base = [
            PYTHON_INTER, main_file_full_path,
            "--action", "base",
            "--out_type", "base",
            "--model_path", args.pretrained_model,
            "--save_path", save_paths['base'],
            "--ft_iter", "0"
            ]
        subprocess.run(base)
    
    elif args.pipeline == 'finetune':
        # only finetune the model for one time.
        finetune = [
            PYTHON_INTER, main_file_full_path,
            "--action", "finetune",
            "--out_type", 'finetune',
            "--model_path", args.pretrained_model,
            "--save_path", save_paths['finetune'],
            "--ft_iter", "1"
        ]
        subprocess.run(finetune)

    elif args.pipeline == "finetune_prune":
        # fine-tune -> prune
        if not check_use_saved_model(args.use_saved_model, save_paths['finetune']):
            # run fine-tune first
            finetune = [
                PYTHON_INTER, main_file_full_path,
                "--action", "finetune",
                "--out_type", "finetune",
                "--model_path", args.pretrained_model,
                "--save_path", save_paths['finetune'],
                "--ft_iter", "1"
            ]
            subprocess.run(finetune)

        for sp in np.arange(0.1, 0.8, 0.1):
            sparsity_txt = "0" + str(sp)[2]
            prune = [
                PYTHON_INTER, main_file_full_path,
                "--action", "prune",
                "--out_type", "finetune_prune",
                "--sparsity", str(sp),
                "--model_path", save_paths['finetune'],
                "--save_path", save_paths["finetune_prune"] + sparsity_txt,
                "--ft_iter", "1"
            ]
            subprocess.run(prune)
    
    elif args.pipeline == "finetune_iter":
        # fine-tune x L
        model_to_finetune = args.pretrained_model
        for p in range(1, args.ft_iter+1):
            finetune = [
                PYTHON_INTER, main_file_full_path,
                "--action", "finetune",
                "--out_type", "finetune_iter",
                "--model_path", model_to_finetune,
                "--save_path", save_paths["finetune_iter"],
                "--ft_iter", str(p)
            ]
            model_to_finetune = save_paths["finetune_iter"]
            subprocess.run(finetune)

    elif args.pipeline == "finetune_iter_prune":
        # (fine-tune) x L -> prune
        if not check_use_saved_model(args.use_saved_model, save_paths['finetune_iter']):
            # run (fine-tune) x L first
            model_to_finetune = args.pretrained_model
            for p in range(1, args.ft_iter+1):
                finetune = [
                    PYTHON_INTER, main_file_full_path,
                    "--action", "finetune",
                    "--out_type", "finetune_iter",
                    "--model_path", model_to_finetune,
                    "--save_path", save_paths["finetune_iter"],
                    "--ft_iter", str(p)
                ]
                model_to_finetune = save_paths["finetune_iter"]
                subprocess.run(finetune)

        for sp in np.arange(0.1, 0.8, 0.1):
            sparsity_txt = "0" + str(sp)[2]
            prune = [
                PYTHON_INTER, main_file_full_path,
                "--action", "prune",
                "--out_type", "finetune_iter_prune",
                "--sparsity", str(sp),
                "--model_path", save_paths["finetune_iter"],
                "--save_path", save_paths["finetune_iter_prune"] + sparsity_txt,
                "--ft_iter", str(args.ft_iter)
            ]
            subprocess.run(prune)
        
    elif args.pipeline == "prune":
        # only prune the model for one time
        for sp in np.arange(0.1, 0.8, 0.1):
            sparsity_txt = "0" + str(sp)[2]
            prune = [
                PYTHON_INTER, main_file_full_path,
                "--action", "prune",
                "--out_type", "prune",
                "--sparsity", str(sp),
                "--model_path", args.pretrained_model,
                "--save_path", save_paths["prune"] + sparsity_txt,
                "--ft_iter", "0"
            ]
            subprocess.run(prune)

    elif args.pipeline == "prune_finetune":
        # prune -> fine-tune
        for sp in np.arange(0.1, 0.8, 0.1):
            sparsity_txt = "0" + str(sp)[2]
            if not check_use_saved_model(args.use_saved_model, save_paths["prune"] + sparsity_txt):
                # run prune first
                prune = [
                    PYTHON_INTER, main_file_full_path,
                    "--action", "prune",
                    "--out_type", "prune",
                    "--sparsity", str(sp),
                    "--model_path", args.pretrained_model,
                    "--save_path", save_paths["prune"] + sparsity_txt,
                    "--ft_iter", "0"
                ]
                subprocess.run(prune)

            finetune = [
                PYTHON_INTER, main_file_full_path,
                "--action", "finetune",
                "--out_type", "prune_finetune",
                "--sparsity", str(sp),
                "--model_path", save_paths["prune"] + sparsity_txt,
                "--save_path", save_paths["prune_finetune"] + sparsity_txt,
                "--ft_iter", "1"
            ]
            subprocess.run(finetune)

    elif args.pipeline == "prune_finetune_iter":
        # prune -> (fine-tune) x L
        for sp in np.arange(0.1, 0.8, 0.1):
            sparsity_txt = "0" + str(sp)[2]
            if not check_use_saved_model(args.use_saved_model, save_paths["prune"] + sparsity_txt):
                # run prune first
                prune = [
                    PYTHON_INTER, main_file_full_path,
                    "--action", "prune",
                    "--out_type", "prune",
                    "--sparsity", str(sp),
                    "--model_path", args.pretrained_model,
                    "--save_path", save_paths["prune"] + sparsity_txt,
                    "--ft_iter", "0"
                ]
                subprocess.run(prune)

            model_to_finetune = save_paths["prune"] + sparsity_txt
            for p in range(1, args.ft_iter+1):
                finetune = [
                    PYTHON_INTER, main_file_full_path,
                    "--action", "finetune",
                    "--out_type", "prune_finetune_iter",
                    "--sparsity", str(sp),
                    "--model_path", model_to_finetune,
                    "--save_path", save_paths["prune_finetune_iter"] + sparsity_txt,
                    "--ft_iter", str(p)
                ]
                model_to_finetune = save_paths["prune_finetune_iter"] + sparsity_txt
                subprocess.run(finetune)

    elif args.pipeline == "iter_pf":
        # (prune -> fine-tune) x L
        model_to_prune = args.pretrained_model
        for p, sp in enumerate(np.arange(0.1, 0.8, 0.1)):
            sparsity_txt = "0" + str(sp)[2]
            prune = [
                PYTHON_INTER, main_file_full_path,
                "--action", "prune",
                "--out_type", "iter_pf",
                "--sparsity", str(sp),
                "--model_path", model_to_prune,
                "--save_path", os.path.join(save_paths["iter_pf"], 'prune' + sparsity_txt),
                "--ft_iter", str(p)
            ]
            subprocess.run(prune)

            finetune = [
                PYTHON_INTER, main_file_full_path,
                "--action", "finetune",
                "--out_type", "iter_pf",
                "--sparsity", str(sp),
                "--model_path", os.path.join(save_paths["iter_pf"], 'prune' + sparsity_txt),
                "--save_path", os.path.join(save_paths["iter_pf"], 'ft' + sparsity_txt),
                "--ft_iter", str(p+1)
            ]
            subprocess.run(finetune)
            model_to_prune = os.path.join(save_paths["iter_pf"], 'ft' + sparsity_txt)

    elif args.pipeline == "iter_fp":
        # (prune -> fine-tune) x L
        model_to_finetune = args.pretrained_model
        for p, sp in enumerate(np.arange(0.1, 0.8, 0.1)):
            sparsity_txt = "0" + str(sp)[2]
            finetune = [
                PYTHON_INTER, main_file_full_path,
                "--action", "finetune",
                "--out_type", "iter_fp",
                "--sparsity", str(sp),
                "--model_path", model_to_finetune,
                "--save_path", os.path.join(save_paths["iter_fp"], 'ft' + sparsity_txt),
                "--ft_iter", str(p+1)
            ]
            subprocess.run(finetune)

            prune = [
                PYTHON_INTER, main_file_full_path,
                "--action", "prune",
                "--out_type", "iter_fp",
                "--sparsity", str(sp),
                "--model_path", os.path.join(save_paths["iter_fp"], 'ft' + sparsity_txt),
                "--save_path", os.path.join(save_paths["iter_fp"], 'prune' + sparsity_txt),
                "--ft_iter", str(p+1)
            ]
            subprocess.run(prune)
            model_to_finetune = os.path.join(save_paths["iter_fp"], 'prune' + sparsity_txt)

    else:
        raise ValueError(f'Unsupported pieline {args.pipeline}')