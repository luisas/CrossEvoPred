process EVALUATE_MODEL {

    input:
    tuple val(meta), model
    tuple val(meta), test_dataset

    output:
    file("*"), emit: evaluation

    script:
    """
    evaluate_model.py --model $model --test_dataset $test_dataset 
    """
}