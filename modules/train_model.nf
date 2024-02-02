process TRAIN_MODEL {

    input:
    tuple val(meta), training_dataset
    tuple val(meta), validation_dataset

    output:
    file("model.pkl"), emit: model

    script:
    """
    train_model.py --training_dataset $training_dataset --validation_dataset $validation_dataset --model_name model.pkl
    """
}