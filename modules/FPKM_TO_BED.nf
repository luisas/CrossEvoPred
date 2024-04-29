process FPKM_TO_BED{
    tag "$meta.id"
    label 'process_medium'

    input:
    tuple val(meta), path(fpkm)

    output:
    tuple val(meta), path(bed), emit: bed

    script:
    prefix = task.ext.prefix ?: "${bed.baseName}"
    """
    fpkm_to_bed.py --fpkm $fpkm \
                   --bed "${prefix}.bed"
    """
}