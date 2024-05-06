process GET_FASTA{
    tag "$meta.id"
    label 'process_medium'

    input:
    tuple val(meta), path(bed)
    tuple val(meta), path(genome), path(fai)

    output:
    tuple val(meta), path("*.fa"), emit: fasta

    script:
    prefix = task.ext.prefix ?: "${bed.baseName}"
    args = task.ext.args ?: ""
    """
    bedtools getfasta -fi $genome -bed $bed -fo ${prefix}.fa $args
    """
}