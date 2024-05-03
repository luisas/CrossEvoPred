process BIG_TO_BEDGRAPH{
    
    input:
    tuple val(meta), path(big)

    output:
    tuple val(meta), path("${big.baseName}.bedgraph"), emit: bedgraph

    script:
    """
    bigWigToBedGraph ${big} ${big.baseName}.bedgraph
    """
}