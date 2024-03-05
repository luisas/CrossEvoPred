process TUNE_MODEL {
    container 'luisas/pytorch_crossevo'    

    input:
    tuple val(meta),  file(training_dataset)
    tuple val(meta2), file(validation_dataset)
    tuple val(meta3), file(tune_config)

    output:
    tuple val(meta), file("${prefix}.yaml"), emit: config

    script:
    prefix = task.ext.prefix ?: "best_config"
    """
    tune.py --training_dataset $training_dataset \
            --validation_dataset $validation_dataset \
            --tune_config $tune_config \
            --out_config "${prefix}.yaml"
    """
}