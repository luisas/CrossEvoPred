#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOW FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
include { validateParameters; paramsHelp; paramsSummaryLog; fromSamplesheet } from 'plugin/nf-validation'
include {   PREPARE_DATA     } from './workflows/prepare_data'
include {   TRAIN_MODEL      } from './modules/train_model'
include {   TUNE_MODEL       } from './modules/tune_model'
include {   EVALUATE_MODEL   } from './modules/evaluate_model'

// Prepare the pipeline parameters
chunk_size = "${params.chunk_size}"



//
// WORKFLOW: Run main nf-core/multiplesequencealign analysis pipeline
//
workflow CROSS_EVO_PRED {


    encode_sheet = Channel.fromSamplesheet('encode_sheet')

    genome = Channel.fromPath("${params.genome}/chr1.fa*").map{
                             it -> [[id:it.parent.baseName], it] }.groupTuple(by:0).map{
                                meta, files -> [meta, files[0], files[1]]
                             }

    config = Channel.fromPath("${params.config}").map{
                                it -> [[id:it.parent.baseName], it] }.groupTuple(by:0)

    // Prepare the data into pytorch dataset objects
    PREPARE_DATA ( genome, chunk_size, encode_sheet )
    training_dataset = PREPARE_DATA.out.train
    validation_dataset = PREPARE_DATA.out.validation
    test_dataset = PREPARE_DATA.out.test

    // take only the first 10 samples for testing
    training_dataset = training_dataset.map{ meta, data -> [meta, data[0..2]]}
    training_dataset.view()

    print("params.tune: ${params.tune}")
    // Train the model
    if( params.tune )  {
        tune_config = Channel.fromPath("${params.tune_config}").map{
                            it -> [[id:it.parent.baseName], it] }.groupTuple(by:0)
        tune_config.view()
        TUNE_MODEL ( training_dataset, validation_dataset, tune_config)
        config = TUNE_MODEL.out.config
    }


    // Merge train and validation 
    // Evaluate the model
    train_and_validation = training_dataset.combine(validation_dataset, by:0 ).map{
                            meta, train, validation -> [meta, train+validation]}
    TRAIN_MODEL ( train_and_validation, config )
    //EVALUATE_MODEL (TRAIN_MODEL.out.model, PREPARE_DATA.out.test)

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