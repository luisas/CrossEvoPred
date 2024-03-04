process SUBSAMPLE_BED{
    
    input:
    tuple val(meta), file(bed)
    val (subsample_size)
    
    output:
    tuple val(meta), file("${prefix}.bed"), emit: bed
    
    script:
    prefix = task.ext.prefix ?: "${bed.baseName}_subsampled"
    """
    shuf -n $subsample_size ${bed} > ${prefix}.bed
    """
}