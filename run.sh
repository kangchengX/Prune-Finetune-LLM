pipelines="prune finetune_prune prune_finetune prune_finetune_iter finetune_iter_prune iter"

for pipeline in $pipelines
do
    python experiments.py --pipeline $pipeline
done