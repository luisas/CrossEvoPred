process EVALUATE_MODEL {
    container 'luisas/pytorch_crossevo'    

    input:
    tuple val(meta), file(model)
    tuple val(meta2), file(test_dataset)

    output:
    tuple val(meta),file("*.csv"), emit: evaluation

    script:
    """
    evaluate.py --model $model --test_dataset $test_dataset --output ${model.baseName}_evaluation.csv
    """
}