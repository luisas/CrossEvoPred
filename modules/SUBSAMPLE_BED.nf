process SUBSAMPLE_BED{
    
    input:
    tuple val(meta), file(bed)
    val (subsample_size)
    
    output:
    tuple val(meta), file("${prefix}_subsampled.bedgraph"), emit: bed
    
    script:
    prefix = task.ext.prefix ?: "${bed.baseName}_subsampled"
    """
    shuf -n $subsample_size ${bed} > ${bed.baseName}_subsampled.bedgraph
    """
}