process CORRECT_BLACKLIST{
        
        input:
        tuple val(meta), file(bedgraph)
        tuple val(meta), file(blacklist_regions)
        
        output:
        tuple val(meta), file("${bedgraph.baseName}_corrected.bedgraph"), emit: bedgraph
        
        script:
        """
        # !/bin/bash
        echo "Starting"

        # calculate 25th percentile of the bedgraph
        percentile=\$(awk '{print \$4}' $bedgraph | sort -n | awk 'BEGIN{c=0} {a[c]=\$1; c++;} END{print a[int((c-1)*0.25)]}')
        echo "Percentile: \$percentile"

        # Sort the blacklist and the bedgraph
        sort -k1,1 -k2,2n $blacklist_regions > sorted_blacklist
        sort -k1,1 -k2,2n $bedgraph > sorted_bedgraph
        echo "Sorted"

        # Get the non-overlapping regions
        bedtools intersect -a sorted_bedgraph -b sorted_blacklist -v > non_overlapping_bedgraph
        echo "Non-overlapping"

        # If the row in bedgraph ovelaps with the blacklist, set the value to the 25th percentile
        bedtools intersect -a sorted_bedgraph -b sorted_blacklist -wa -wb | awk -v p=\$percentile '{\$4=p}1' > corrected_bedgraph
        echo "Corrected"
        
        # Merge, sort and save
        cat non_overlapping_bedgraph corrected_bedgraph | sort -k1,1 -k2,2n > ${bedgraph.baseName}_corrected.bedgraph
        """
}