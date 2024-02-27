process COBALT_SPLIT{

    container 'luisas/hhmer'    

    input:
    tuple val(meta), path(stockholm_alignment)

    output:
    tuple val(meta), file("*_train.bed"), emit: train
    tuple val(meta), file("*_validation.bed"), emit: validation
    tuple val(meta), file("*_test.bed"), emit: test

    script:
    prefix = task.ext.prefix ?: "${stockholm_alignment.baseName}"
    """
    ./create-profmark --onlysplit ${prefix} ${stockholm_alignment}
    """

}