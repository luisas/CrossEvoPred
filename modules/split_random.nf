process SPLIT_RANDOM{

    container 'luisas/pytorch_crossevo'    

    input:
    tuple val(meta), path(genome), path(fai)
    val(chunk_size)

    output:
    tuple val(meta), file("*_train.bed"), emit: train
    tuple val(meta), file("*_validation.bed"), emit: validation
    tuple val(meta), file("*_test.bed"), emit: test

    script:
    """
    split_random.py -i $genome -o ${genome.baseName}.bed -l $chunk_size
    """

}