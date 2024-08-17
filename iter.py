import subprocess
import os
from utils import get_last_sparsity_iter


cache_path = './hf_cache/'
os.environ['HF_HOME']=cache_path
os.environ['TRANSFORMERS_CACHE'] = cache_path


if __name__ == "__main__":
    current_directory = os.getcwd()

    sparsity = 0.7
    ###PRUNE FINETUNE:
    print("*"*20 + f" PRUNE FT, SPARSITY:{sparsity} "+"*"*20)
    iterations = 7
    k = sparsity/iterations
    curr = 0.23#Change this back to 1

    curr_str = "0_77"
    for i in range(6,iterations+1):# Change this back to 1
        if i > 1:
            model_path = f"{os.path.join(current_directory, f'models/iter/prune_ft{curr_str}')}"
        else:
            model_path = "baffo32/decapoda-research-llama-7B-hf"
        sp = k/curr
        curr = curr - sp*curr
        sp = k*i
        print("current: ", curr*100, sp)
        sparsity_txt = "0" + str(sp)[2:5]
        curr_str = f"{(1-curr):.2f}".replace('.', '_')[:5]
        print("PRUNING TO SPARSITY:", sp)
        print("Current model path", model_path)
        print("Current save path", f'models/iter/prune{curr_str}')
        prune = [
            "/cs/student/projects3/COMP0087/grp1/.venv/bin/python", "/cs/student/projects3/COMP0087/grp1/main.py",
            "--action", "prune",
            "--out_type", "iter",
            "--sparsity", f"{sp}",
            "--model_path", model_path,
            "--save_path", f"{os.path.join(current_directory, f'models/iter/prune{curr_str}')}",
            "--epochs","0.1",
            "--i", str(i+1)
        ]
        subprocess.run(prune)
        print("starting finetune...")
        finetune = [
            "/cs/student/projects3/COMP0087/grp1/.venv/bin/python", "/cs/student/projects3/COMP0087/grp1/main.py",
            "--action", "finetune",
            "--out_type", "iter",
            "--check_sparsity", "True",
            "--epochs","0.1",
            "--sparsity", f"{sp}",
            "--model_path", f"{os.path.join(current_directory, f'models/iter/prune{curr_str}')}",
            "--save_path", f"{os.path.join(current_directory, f'models/iter/prune_ft{curr_str}')}",
            "--i", str(i+1)
        ]
        subprocess.run(finetune)
        curr = get_last_sparsity_iter()
