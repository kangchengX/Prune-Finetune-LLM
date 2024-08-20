import subprocess, os, argparse
import numpy as np
from constant import PIPELINE_SAVE_PATH_MAP, PYTHON_INTER


cache_path = './hf_cache/'
os.environ['HF_HOME'] = cache_path
os.environ['TRANSFORMERS_CACHE'] = cache_path


def check_use_saved_model(use_saved_model: bool, saved_model_path: str):
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
        "iter": (prune -> fine-tune) x L.
        """
    )
    parser.add_argument('--ft_iter', type=int, help='Number of iterations to run finetune', default=5)
    parser.add_argument(
        '--use_saved_model', 
        type=str, 
        default='True', 
        help="""If True, some fine-tuning or pruning in one pipeline will be executed based on \
        the saved models from the previous executed pipelines.
        """
    )
    parser.add_argument('--pretrained_model', type=str, default="baffo32/decapoda-research-llama-7B-hf")

    args = parser.parse_args()
    parser.use_saved_model = eval(parser.use_saved_model)
    current_file_directory = os.path.dirname(__file__)
    main_file_full_path = os.path.join(current_file_directory, 'main.py')
    save_paths = {pro: os.path.join(current_file_directory, path) for pro, path in PIPELINE_SAVE_PATH_MAP.keys()}

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
        subprocess.run(base, check=True)
    
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
            subprocess.run(prune, check=True)
    
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

        prune = [
            PYTHON_INTER, main_file_full_path,
            "--action", "prune",
            "--out_type", "finetune_iter_prune",
            "--sparsity", str(0.5),
            "--model_path", save_paths["finetune_iter"],
            "--save_path", save_paths["finetune_iter_prune"]
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
                prune = [
                    PYTHON_INTER, main_file_full_path,
                    "--action", "prune",
                    "--out_type", "prune",
                    "--sparsity", str(sp),
                    "--model_path", args.pretrained_model,
                    "--save_path", save_paths["prune"] + sparsity_txt,
                    "--ft_iter", "0"
                ]
            subprocess.run(prune, check=True)
            finetune = [
                PYTHON_INTER, main_file_full_path,
                "--action", "finetune",
                "--out_type", "prune_finetune",
                "--sparsity", str(sp),
                "--model_path", save_paths["prune"] + sparsity_txt,
                "--save_path", save_paths["prune_finetune"] + sparsity_txt,
                "--ft_iter", "1"
            ]
            subprocess.run(finetune, check=True)

    elif args.pipeline == "prune_finetune_iter":
        # prune -> (fine-tune) x L
        for sp in np.arange(0.1, 0.8, 0.1):
            sparsity_txt = "0" + str(sp)[2]
            if not check_use_saved_model(args.use_saved_model, save_paths["prune"] + sparsity_txt):
                prune = [
                    PYTHON_INTER, main_file_full_path,
                    "--action", "prune",
                    "--out_type", "prune",
                    "--sparsity", str(sp),
                    "--model_path", args.pretrained_model,
                    "--save_path", save_paths["prune"] + sparsity_txt,
                    "--ft_iter", "0"
                ]
            subprocess.run(prune, check=True)

            for p in range(1, args.ft_iter+1):
                model_to_finetune = save_paths["prune"] + sparsity_txt
                finetune = [
                    PYTHON_INTER, main_file_full_path,
                    "--action", "finetune",
                    "--out_type", "prune_finetune_iter",
                    "--sparsity", str(sp),
                    "--model_path", model_to_finetune,
                    "--save_path", save_paths["prune_finetune_iter"],
                    "--ft_iter", str(p)
                ]
                model_to_finetune = save_paths["prune_finetune_iter"]
                subprocess.run(finetune, check=True)

    elif args.pipeline == "iter":
        # (prune -> fine-tune) x L
        model_to_prune = args.pretrained_model
        for p, sp in enumerate(np.arange(0.1, 0.8, 0.1)):
            sparsity_txt = "0" + str(sp)[2]
            prune = [
                PYTHON_INTER, main_file_full_path,
                "--action", "prune",
                "--out_type", "iter",
                "--sparsity", str(sp),
                "--model_path", model_to_prune,
                "--save_path", os.path.join(save_paths['iter'], 'prune' + sparsity_txt),
                "--ft_iter", str(p),
                '--sparsity_per_iter', str(0.1)
            ]
            subprocess.run(prune)
            finetune = [
                PYTHON_INTER, main_file_full_path,
                "--action", "finetune",
                "--out_type", "iter",
                "--sparsity", f"{sp}",
                "--model_path", os.path.join(save_paths['iter'], 'prune' + sparsity_txt),
                "--save_path", os.path.join(save_paths['iter'], 'ft' + sparsity_txt),
                "--ft_iter", str(p+1)
            ]
            subprocess.run(finetune)
            model_to_prune = os.path.join(save_paths['iter'], 'ft' + sparsity_txt)
    else:
        raise ValueError(f'Unsupported pieline {args.pipeline}')