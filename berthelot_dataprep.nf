#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOW FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
include { validateParameters; paramsHelp; paramsSummaryLog; fromSamplesheet } from 'plugin/nf-validation'
include { BERTHELOT_DATAPREP } from './workflows/BERTHELOT_DATAPREP.nf'
include { PREPROCESS_DATA    } from './subworkflows/preprocess_data.nf'

workflow DATA_PREP {

    genome  = Channel.fromSamplesheet('genomes')         
    bigwigs = Channel.fromSamplesheet('bigwigs')
    
    genome.view()
    bedgraph = bigwigs
    // PREPROCESS_DATA(bigwigs)

    // bedgraph = PREPROCESS_DATA.out.bedgraph

    bedgraph.view()
    // merge genome and bigwigs by species in meta
    genome.map{ meta, genome, index -> [ meta["species"], meta, genome, index ] }
           .combine(bedgraph.map{ meta, bigwig -> [ meta["species"], meta, bigwig ] }, by:0 ).view()
           .map{ species, meta_genome, genome, index, meta_file, bigwig -> [species, bigwig, genome, index] }.view() 
    

    // genome.view()
    
    //BERTHELOT_DATAPREP()


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