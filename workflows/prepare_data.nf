
include{ FETCH_DATA               } from '../subworkflows/fetch_data'
include{ SPLIT_DATA               } from '../subworkflows/split_data'
include{ PREPROCESS_DATA          } from '../subworkflows/preprocess_data'
include{ CREATE_DATASET_OBJECT   as CREATE_DATASET_OBJECT_TRAIN;
         CREATE_DATASET_OBJECT   as CREATE_DATASET_OBJECT_VALIDATION; 
         CREATE_DATASET_OBJECT   as CREATE_DATASET_OBJECT_TEST } from '../modules/create_dataset_object'
include { PREP_INPUT_FILE as PREP_INPUT_FILE_TRAIN; 
          PREP_INPUT_FILE as PREP_INPUT_FILE_VALIDATION; 
          PREP_INPUT_FILE as PREP_INPUT_FILE_TEST } from '../modules/prep_input_file'

workflow PREPARE_DATA {

    take: 
    genome
    chunk_size
    encode_sheet
    
    main: 

    // Download and preprocess the data
    if (params.fetch_data){
        // Fetch the data
        FETCH_DATA (encode_sheet)
        functional_data = PREPROCESS_DATA(FETCH_DATA.out.data)

        // Split dataset into train, test, and validation sets
        // and extract the corresponding fasta files
        SPLIT_DATA(genome, chunk_size)
        train_fasta      = PREP_INPUT_FILE_TRAIN(SPLIT_DATA.out.train, functional_data, genome, "${params.bin_size}", "mean")
        validation_fasta = PREP_INPUT_FILE_VALIDATION(SPLIT_DATA.out.validation, functional_data, genome, "${params.bin_size}", "mean")
        test_fasta       = PREP_INPUT_FILE_TEST(SPLIT_DATA.out.test, functional_data, genome, "${params.bin_size}", "mean")    
        
        // Create dataset objects
        train      = CREATE_DATASET_OBJECT_TRAIN(train_fasta)
        validation = CREATE_DATASET_OBJECT_VALIDATION(validation_fasta)
        test       = CREATE_DATASET_OBJECT_TEST(test_fasta)

    }else{
        // Small test file (this mode is for testing purposes only)
        functional_data = Channel.fromPath("${params.bedgraph_test_file}").map{it -> [[:], it]}
        test_bed_file = Channel.fromPath("${params.test_bed_file}").map{it -> [[:], it]}
        CREATE_DATASET_OBJECT( test_bed_file, genome, functional_data, "${params.bin_size}")
        train =  CREATE_DATASET_OBJECT.out.object
        validation = CREATE_DATASET_OBJECT.out.object
        test = CREATE_DATASET_OBJECT.out.object
    } 

    emit: 
    train 
    validation 
    test 


}