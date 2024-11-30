import sys

PIPELINE_SAVE_PATH_MAP = {
    "base" : "base",
    "finetune" : "model/ft",
    "finetune_prune" : "models/ft_prune",
    "finetune_iter" : "models/ft_iter",
    "finetune_iter_prune" : "models/ft_iter_prune",
    "prune" : "model/prune",
    "prune_finetune" : "models/prune_ft",
    "prune_finetune_iter" : "models/prune_ft_iter",
    "iter_fp" : "models/iter_fp",
    "iter_pf" : "models/iter_pf"
}

PYTHON_INTER = sys.executable