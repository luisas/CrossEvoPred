process MERGE_BEDS_WITH_SPLIT_VALUE{

    input: 
    tuple val(meta), path(training_bed)
    tuple val(meta1), path(validation_bed)
    tuple val(meta2), path(test_bed)

    output:
    tuple val(meta), path("${prefix}.bed"), emit: bed

    script:
    prefix = args.prefix ? args.prefix : training_bed.getBaseName().toString().replaceAll("_train", "")
    """
    # append a column to the bed file with the split value ( 0 for training, 1 for validation, 2 for test)
    cat ${training_bed} | awk -v OFS='\t' '{print \$0, 0}' > ${training_bed}.tmp
    cat ${validation_bed} | awk -v OFS='\t' '{print \$0, 1}' > ${validation_bed}.tmp
    cat ${test_bed} | awk -v OFS='\t' '{print \$0, 2}' > ${test_bed}.tmp
    cat ${training_bed}.tmp ${validation_bed}.tmp ${test_bed}.tmp > ${prefix}.bed
    """
}