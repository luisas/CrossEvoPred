process CLIP_BEDGRAPH{

    input:
    tuple val(meta), path(bedgraph)
    val(clip_threshold)

    output:
    tuple val(meta), path("${prefix}.bedgraph"), emit: bedgraph

    script:
    prefix = task.ext.prefix ?: "${bedgraph.baseName}_clipped"
    """
    # !/bin/bash
    awk -v tc="$clip_threshold" 'function min(x, y) {return (x < y) ? x : y} 
                                function max(x, y) {return (x > y) ? x : y} 
                                {print \$1"\t"\$2"\t"\$3"\t"min(\$4, tc + sqrt(max(0, \$4 - tc)))}' "$bedgraph" > "${prefix}.bedgraph"    
    """

}