import subprocess, os, argparse


cache_path = './hf_cache/'
os.environ['HF_HOME']=cache_path
os.environ['TRANSFORMERS_CACHE'] = cache_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--procedure', type=str, help='Procedure to run', default="base")
    parser.add_argument('--num_iter', type=int, help='Number of iterations to run finetune', default=5)
    args = parser.parse_args()
    current_directory = os.getcwd()
    
    if args.procedure == "finetune":
        base_model = "baffo32/decapoda-research-llama-7B-hf"
        for p in range(1,args.num_iter+1):
            finetune = [
                    "/cs/student/projects3/COMP0087/grp1/.venv/bin/python", "/cs/student/projects3/COMP0087/grp1/main.py",
                    "--action", "finetune",
                    "--out_type", "finetune",
                    "--model_path", base_model,
                    "--save_path", f"./models/ft_iter",
                    "--check_sparsity", "True",
                    "--eval_ppl", "True",
                    "--epochs","0.1",
                    "--i", str(p)
                ]
            base_model = "./models/ft_iter"
            subprocess.run(finetune)
        #We add a prune at the end (FINETUNE ITER PRUNE)
        prune = [
                "/cs/student/projects3/COMP0087/grp1/.venv/bin/python", "/cs/student/projects3/COMP0087/grp1/main.py",
                "--action", "prune",
                "--out_type", "finetune_iter_prune",
                "--model_path", base_model,
                "--sparsity", f"{0.5}",
                "--save_path", f"./models/ft_iter_prune",
                "--check_sparsity", "True",
                "--eval_ppl", "True",
                "--epochs","0.1",
            ]
        subprocess.run(prune)
    elif args.procedure == "prune":
        base_model = "baffo32/decapoda-research-llama-7B-hf"
        for sp in [.1,.2,.3,.4,.5,.6,.7]:
            prune = [
                    "/cs/student/projects3/COMP0087/grp1/.venv/bin/python", "/cs/student/projects3/COMP0087/grp1/main.py",
                    "--action", "prune",
                    "--out_type", "prune",
                    "--model_path", base_model,
                    "--sparsity", f"{sp}",
                    "--save_path", f"./models/prune",
                    "--check_sparsity", "True",
                    "--eval_ppl", "True",
                    "--epochs","0.1",
                ]
            subprocess.run(prune)
    elif args.procedure == "finetune_prune":
        ###FINETUNE PRUNE:
        print("*"*20 + f" Finetuning first: "+"*"*20)
        finetune = [
                "/cs/student/projects3/COMP0087/grp1/.venv/bin/python", "/cs/student/projects3/COMP0087/grp1/main.py",
                "--action", "finetune",
                "--out_type", "finetune",
                "--model_path", "baffo32/decapoda-research-llama-7B-hf",
                "--save_path", f"{os.path.join(current_directory, 'models/ft')}",
                "--epochs","0.1",
            ]
        # subprocess.run(finetune)
        for sp in [.3,.4,.5,.6,.7]:
            sparsity_txt = "0" + str(sp)[2:]
            print("*"*20 + f" FT PRUNE, SPARSITY:{sp} "+"*"*20)
            prune = [
                "/cs/student/projects3/COMP0087/grp1/.venv/bin/python", "/cs/student/projects3/COMP0087/grp1/main.py",
                "--action", "prune",
                "--out_type", "finetune_prune",
                "--sparsity", f"{sp}",
                "--model_path", f"{os.path.join(current_directory, 'models/ft')}",
                "--save_path", f"{os.path.join(current_directory, f'models/ft_prune')}"
            ]
            subprocess.run(prune, check=True)
    elif args.procedure == "prune_finetune":
        for sp in [.5]:#[.1,.2,.3,.4,.5,.6,.7]:
            sparsity_txt = "0" + str(sp)[2:]
            ###PRUNE FINETUNE:
            print("*"*20 + f" PRUNE FT, SPARSITY:{sp} "+"*"*20)
            model_output = f"{os.path.join(current_directory, f'models/prune{sparsity_txt}')}"
            prune = [
                "/cs/student/projects3/COMP0087/grp1/.venv/bin/python", "/cs/student/projects3/COMP0087/grp1/main.py",
                "--action", "prune",
                "--out_type", "prune",
                "--sparsity", f"{sp}",
                "--model_path", "baffo32/decapoda-research-llama-7B-hf",
                "--save_path", model_output
            ]
            subprocess.run(prune, check=True)
            finetune = [
                "/cs/student/projects3/COMP0087/grp1/.venv/bin/python", "/cs/student/projects3/COMP0087/grp1/main.py",
                "--action", "finetune",
                "--out_type", "prune_finetune",
                "--sparsity", f"{sp}",
                "--model_path", model_output,
                "--save_path", f"{os.path.join(current_directory, f'models/prune_ft{sparsity_txt}')}",
                "--i", str(1)
            ]
            subprocess.run(finetune, check=True)
    elif args.procedure == "prune_ft_iter":
        for sp in [.5]:#[.1,.2,.3,.4,.5,.6,.7]:
            sparsity_txt = "0" + str(sp)[2:]
            ###PRUNE FINETUNE:
            print("*"*20 + f" PRUNE FT, SPARSITY:{sp} "+"*"*20)
            model_output = f"{os.path.join(current_directory, f'models/prune{sparsity_txt}')}"
            prune = [
                "/cs/student/projects3/COMP0087/grp1/.venv/bin/python", "/cs/student/projects3/COMP0087/grp1/main.py",
                "--action", "prune",
                "--out_type", "prune",
                "--sparsity", f"{sp}",
                "--model_path", "baffo32/decapoda-research-llama-7B-hf",
                "--save_path", model_output
            ]
            subprocess.run(prune, check=True)
            for p in range(1,args.num_iter+1):
                finetune = [
                    "/cs/student/projects3/COMP0087/grp1/.venv/bin/python", "/cs/student/projects3/COMP0087/grp1/main.py",
                    "--action", "finetune",
                    "--out_type", "prune_finetune_iter",
                    "--sparsity", f"{sp}",
                    "--model_path", model_output,
                    "--save_path", f"{os.path.join(current_directory, f'models/prune_ft_iter')}",
                    "--epochs","0.1",
                    "--i", str(p)
                ]
                model_output = f"{os.path.join(current_directory, f'models/prune_ft_iter')}"
                subprocess.run(finetune, check=True)
    else:
        ##Base Model:
        base = [
                "/cs/student/projects3/COMP0087/grp1/.venv/bin/python","/cs/student/projects3/COMP0087/grp1/main.py",
                "--action", "base",
                "--out_type", "base",
                "--model_path", f"baffo32/decapoda-research-llama-7B-hf",
                "--save_path", f"baffo32/decapoda-research-llama-7B-hf",
                "--check_sparsity", "True",
                "--eval_ppl", "True",
                "--i", "0"
            ]
        subprocess.run(base, check=True)