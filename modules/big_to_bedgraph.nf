process BIG_TO_BEDGRAPH{
    
    input:
    tuple val(meta), file(big)

    output:
    tuple val(meta), file("${big.baseName}.bedgraph"), emit: bedgraph

    script:
    """
    bigWigToBedGraph ${big} ${big.baseName}.bedgraph
    """
}