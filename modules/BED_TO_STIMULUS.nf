process BED_TO_STIMULUS{
    tag "$meta.id"

    input:
    tuple val(meta), path(bed)

    output:
    tuple val(meta), path("${prefix}.csv"), emit: csv

    script:
    prefix = task.ext.prefix ?: "${bed.baseName}"
    """
    bed_to_stimulus.py --bed $bed \
                       --species $meta.species \
                       --out ${prefix}.csv
    """
}