process SUBSET_FOR_TESTING{
    
    input:
    tuple val(meta), file(bedgraph)
    
    output:
    tuple val(meta), file("${bedgraph.baseName}_subset.bedgraph"), emit: bedgraph
    
    script:
    """
    head -n 3000 ${bedgraph} > ${bedgraph.baseName}_subset.bedgraph
    """
}