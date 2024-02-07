process TRAIN_MODEL {
    container 'luisas/pytorch_crossevo'    

    input:
    tuple val(meta),  file(training_dataset)
    tuple val(meta2), file(config)
    //tuple val(meta2), file(validation_dataset)

    output:
    tuple val(meta), file("model.pkl"), emit: model

    script:
    """
    train.py --training_dataset $training_dataset --config $config --model_name "model.pkl"
    """
}