process CHUNK_GENOME{

    container 'luisas/pytorch_crossevo'    

    input:
    tuple val(meta), path(genome), path(fai)
    val(chunk_size)

    output:
    tuple val(meta), file("*_chunks.bed"), emit: bed

    script:
    """
    chunk_genome.py -i $genome -o ${genome.baseName}_chunks.bed -l $chunk_size
    """

}