#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOW FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
include { validateParameters; paramsHelp; paramsSummaryLog; fromSamplesheet } from 'plugin/nf-validation'
include { STIMULUS_DATAPREP } from './workflows/STIMULUS_DATAPREP.nf'
include{ SPLIT_DATA               } from './subworkflows/split_data'
include { MERGE_BEDS_WITH_SPLIT_VALUE } from './modules/MERGE_BEDS_WITH_SPLIT_VALUE'
workflow DATA_PREP {

    genome  = Channel.fromSamplesheet('genomes')         
    bigwigs = Channel.fromSamplesheet('bigwigs')
    
    
    // if bed file if given, use it, otherwise make it random
    if (params.beds){
        bed     = Channel.fromSamplesheet('beds')
    } else {
        SPLIT_DATA(genome, params.sequence_length)

        MERGE_BEDS_WITH_SPLIT_VALUE(SPLIT_DATA.out.train, SPLIT_DATA.out.validation, SPLIT_DATA.out.test)

        bed = MERGE_BEDS_WITH_SPLIT_VALUE.out.bed
    }


    // merge genome and bigwigs by species in meta
    ch_bedgraph_and_genome = genome.map{ meta, genome, index -> [ meta["species"], meta, genome, index ] }
                            .combine(bigwigs.map{ meta, bigwig -> [ meta["species"], meta, bigwig ] }, by:0 )
                            .map{ species, meta_genome, genome, index, meta_file, bigwig -> [["species": species], bigwig, genome, index] }

    STIMULUS_DATAPREP(ch_bedgraph_and_genome, bed)


}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN ALL WORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow {

    DATA_PREP ()

}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/