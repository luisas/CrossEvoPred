#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOW FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
include { validateParameters; paramsHelp; paramsSummaryLog; fromSamplesheet } from 'plugin/nf-validation'
include {   PREPARE_DATA  } from './workflows/prepare_data'


// Prepare the pipeline parameters
chunk_size = "${params.chunk_size}"


//
// WORKFLOW: Run main nf-core/multiplesequencealign analysis pipeline
//
workflow CROSS_EVO_PRED {


    encode_sheet = Channel.fromSamplesheet('encode_sheet')
    
    genome = Channel.fromPath("${params.genome}/chr1.fa").map{
                             it -> [[id:it.parent.baseName], it] }.groupTuple(by:0)
    

    // Prepare the data into pytorch dataset objects
    PREPARE_DATA ( genome, chunk_size, encode_sheet)

    // Train the model
    TRAIN_MODEL ( PREPARE_DATA.out.train, PREPARE_DATA.out.validation)

    // Evaluate the model
    EVALUATE_MODEL (TRAIN_MODEL.out.model, PREPARE_DATA.out.test)

}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN ALL WORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// WORKFLOW: Execute a single named workflow for the pipeline
// See: https://github.com/nf-core/rnaseq/issues/619
//
workflow {

    CROSS_EVO_PRED ()

}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/