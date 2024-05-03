#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOW FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
include { validateParameters; paramsHelp; paramsSummaryLog; fromSamplesheet } from 'plugin/nf-validation'
include { STIMULUS_DATAPREP } from './workflows/STIMULUS_DATAPREP.nf'

workflow DATA_PREP {

    genome  = Channel.fromSamplesheet('genomes')         
    bigwigs = Channel.fromSamplesheet('bigwigs')
    
    // merge genome and bigwigs by species in meta
    ch_bedgraph_and_genome = genome.map{ meta, genome, index -> [ meta["species"], meta, genome, index ] }
                            .combine(bigwigs.map{ meta, bigwig -> [ meta["species"], meta, bigwig ] }, by:0 )
                            .map{ species, meta_genome, genome, index, meta_file, bigwig -> [["species": species], bigwig, genome, index] }

    STIMULUS_DATAPREP(ch_bedgraph_and_genome)


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