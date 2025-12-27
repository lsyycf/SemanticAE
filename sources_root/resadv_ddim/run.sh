

export PYTHONPATH=./:../:../../
# Abstracted label generation
python evaluation/test_attack_coarse.py -n semanticae -c insur/base -i run_abstracted  --task.dev_run=False

python ../image_evaluation/evaluator.py -n semanticae -c eval/default_evaluation_multilabel -i test \
        --evaluation.dataset.images_path="\${PATH_LOGS}/semanticae/insur/base/run_abstracted/out_images" \
        --evaluation.dataset.label_path="\${PATH_LOGS}/semanticae/insur/base/run_abstracted/labels.txt" \
        --evaluation.dataset.anchor_images_path="\${PATH_LOGS}/semanticae/insur/base/run_abstracted/out_images_exemplar" \
        --evaluation.dataset.label_start=0

# results stored at logs/default_evaluation_multilabel/test/results/val_results.csv

# Refined label generation
# python evaluation/test_attack.py -n semanticae -c insur/base -i run --task.rerun=True --task.dev_run=False
#
# Evaluation
#python ../image_evaluation/evaluator.py -n semanticae -c eval/default_evaluation -i test \
#        --evaluation.dataset.images_path="\${PATH_LOGS}/semanticae/insur/base/run/out_images" \
#        --evaluation.dataset.label_path="\${PATH_LOGS}/semanticae/insur/base/run/labels.txt" \
#        --evaluation.dataset.anchor_images_path="\${PATH_LOGS}/semanticae/insur/base/run/out_images_exemplar" \
#        --evaluation.dataset.label_start=0
# results stored at logs/default_evaluation/test/results/val_results.csv



