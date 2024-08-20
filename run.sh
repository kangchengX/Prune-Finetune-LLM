pipelines="prune finetune_prune prune_finetune prune_finetune_iter iter"

for pipeline in $pipelines
do
    python experiments.py --pipeline $pipeline
done