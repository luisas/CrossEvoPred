process REMOVE_BLACKLIST{

    container "biocontainers/bedtools:v2.27.0_cv3"

    input:
    tuple val(meta), file(bed)
    tuple val(meta), file(blacklist)

    output:
    tuple val(meta), file("${prefix}_no_blacklist.bed")

    script:
    prefix = task.ext.prefix ?: "${bed.baseName}"
    """
    #!/bin/bash

    # Remove entries from bed file if they overlap by more than 50% with the blacklist regions
    bedtools intersect -a $bed -b $blacklist -f 0.5 -v > ${prefix}_no_blacklist.bed
    """
}