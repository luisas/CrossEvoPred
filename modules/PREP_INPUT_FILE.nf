process PREP_INPUT_FILE{

    container "biocontainers/bedtools:v2.27.0_cv2"
    label "process_medium"

    input:
    tuple val(meta2), file(bed)
    tuple val(meta), file(bedgraph)
    tuple val(meta3), file(genome), file(fai)
    val(window_size)
    val(metric)
    val(coverage_threshold)

    output:
    tuple val(meta), file("${bed.baseName}-sequences-and-labels.bed"), emit: training_file
  
    script:
    """
    # !/bin/bash
    echo "Starting"
    
    # Calculate the max value for each window
    # sort bed 
    bedtools sort -i $bed > ${bed.baseName}_sorted.bed
    bedtools map -a  ${bed.baseName}_sorted.bed -b $bedgraph -c 4 -o max > max.bedgraph
    awk -v threshold=$coverage_threshold '{if (\$4 < threshold) print \$0}' max.bedgraph > not_enough_max.bedgraph
    bedtools intersect -a ${bed.baseName}_sorted.bed -b not_enough_max.bedgraph -v > ${bed.baseName}_coverage_filtered.bed    
    echo "max value calculated"


    # Create windows in the original bed file
    bedtools makewindows -b ${bed.baseName}_coverage_filtered.bed -w $window_size | bedtools sort -i - > ${bed.baseName}_windows_sorted.bed
    echo "windows created"


    # Map begraph to the windows
    bedtools map -a ${bed.baseName}_windows_sorted.bed -b $bedgraph -c 4 -o sum > bed_windowed.bedgraph
    echo "bedgraph mapped to windows"


    # Map original bed to the windowed bedgraph 
    bedtools sort -i ${bed.baseName}_coverage_filtered.bed > ${bed.baseName}_coverage_filtered_sorted.bed
    bedtools map -a ${bed.baseName}_coverage_filtered_sorted.bed -b bed_windowed.bedgraph -c 4 -o collapse > ${bed.baseName}_labels.bed
    echo "original bed mapped to windowed bedgraph"

    # -----------------------------------------------------------
    # Use the extracted informations to create the training file
    # -----------------------------------------------------------
    bedtools getfasta -fi $genome -bed ${bed.baseName}_labels.bed -bedOut > ${bed.baseName}-sequences-and-labels.bed  
    """
}