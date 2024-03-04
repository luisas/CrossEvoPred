process PREP_INPUT_FILE{

    container "biocontainers/bedtools:v2.27.0_cv3"
    label "process_medium"

    input:
    tuple val(meta), file(bed)
    tuple val(meta2), file(bedgraph)
    tuple val(meta), file(genome), file(fai)
    val(window_size)
    val(metric)

    output:
    //tuple val(meta2), file("${bed.baseName}_sequences_and_labels.bed"), emit: training_file
    tuple val(meta2), file("${bed.baseName}-sequences-and-labels.bed"), emit: training_file
  
    script:
    """
    # !/bin/bash
    echo "Starting"

    # Create windows in the original bed file
    bedtools makewindows -b $bed -w $window_size > ${bed.baseName}_windows.bed
    bedtools sort -i ${bed.baseName}_windows.bed > ${bed.baseName}_windows_sorted.bed
    echo "windows created"

    # Map begraph to the windows
    bedtools map -a ${bed.baseName}_windows_sorted.bed -b $bedgraph -c 4 -o sum > bed_windowed.bedgraph
    # substitute . for 0
    sed -i 's/\./0/g' bed_windowed.bedgraph
    echo "bedgraph mapped to windows"

    # Map original bed to the windowed bedgraph 
    bedtools sort -i $bed > ${bed.baseName}_sorted.bed
    bedtools map -a ${bed.baseName}_sorted.bed -b bed_windowed.bedgraph -c 4 -o collapse > ${bed.baseName}_labels.bed
    echo "original bed mapped to windowed bedgraph"

    # -----------------------------------------------------------
    # Use the extracted informations to create the training file
    # -----------------------------------------------------------
    bedtools getfasta -fi $genome -bed ${bed.baseName}_labels.bed -bedOut > ${bed.baseName}-sequences-and-labels.bed  
    """
}