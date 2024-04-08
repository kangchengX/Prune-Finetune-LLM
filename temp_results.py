import subprocess
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--procedure', type=str, help='Procedure to run', default="prune_finetune")
    parser.add_argument('--num_iter', type=int, help='Number of iterations to run finetune', default=1)
    args = parser.parse_args()
    current_directory = os.getcwd()

    """if args.procedure == "finetune_prune":
        for p in range(1,8):
            finetune_prune = [
                    "/cs/student/projects3/COMP0087/grp1/.venv/bin/python", "/cs/student/projects3/COMP0087/grp1/main.py",
                    "--action", "base",
                    "--out_type", "finetune_prune",
                    "--model_path", f"{os.path.join(current_directory, f'models/ft_prune0{p}')}",
                    "--save_path", f"{os.path.join(current_directory, f'models/ft_prune0{p}')}",
                    "--train", "False",
                    "--epochs","0.1",
                    "--i", str(p)
                ]
            subprocess.run(finetune_prune)"""
    if args.procedure == "iter":
        directories = []
        path = os.path.join(current_directory, f'models/iter/')
        # List all files and directories in the specified path
        for item in os.listdir(path):
            # Check if the item is a directory
            if os.path.isdir(os.path.join(path, item)):
                directories.append(item)
        for p,direct in enumerate(directories):
            iter = [
                    "/cs/student/projects3/COMP0087/grp1/.venv/bin/python", "/cs/student/projects3/COMP0087/grp1/main.py",
                    "--action", "base",
                    "--out_type", "iter",
                    "--model_path", f"{os.path.join(current_directory, f'models/iter/{direct}')}",
                    "--save_path", f"{os.path.join(current_directory, f'models/iter/{direct}')}",
                    "--train", "False",
                    "--epochs","0.1",
                    "--i", str(p+1)
                ]
            subprocess.run(iter)
    if args.procedure == "prune_finetune":
        for p in range(1,8):
            prune_finetune = [
                    "/cs/student/projects3/COMP0087/grp1/.venv/bin/python", "/cs/student/projects3/COMP0087/grp1/main.py",
                    "--action", "base",
                    "--out_type", "prune_finetune",
                    "--model_path", f"{os.path.join(current_directory, f'models/prune_ft0{p}')}",
                    "--save_path", f"{os.path.join(current_directory, f'models/prune_ft0{p}')}",
                    "--train", "False",
                    "--epochs","0.1",
                    "--i", str(p)
                ]
            subprocess.run(prune_finetune)