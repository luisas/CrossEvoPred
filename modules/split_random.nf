process SPLIT_RANDOM{

    input:
    tuple val(meta), path(genome)
    val(chunk_size)

    output:
    tuple val(meta), file("*.bed")

    script:
    """
    split_random.py -i $genome -o ${genome.baseName}.bed -l $chunk_size
    """

}