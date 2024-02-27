process TRAIN_MODEL {
    container 'luisas/pytorch_crossevo'    

    input:
    tuple val(meta),  file(training_dataset)
    tuple val(meta2), file(config)

    output:
    tuple val(meta), file("*.pkl"), emit: model

    script:
    prefix = task.ext.prefix ?: "model"
    """
    train.py --training_dataset $training_dataset --config $config --model_name "${prefix}.pkl" --trainer_name "${prefix}_trainer.pkl"
    """
}