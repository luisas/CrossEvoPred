
include{ FETCH_DATA               } from '../subworkflows/fetch_data'
include{ SPLIT_DATA               } from '../subworkflows/split_data'
include{ PREPROCESS_DATA          } from '../subworkflows/preprocess_data'
include{ CREATE_DATASET_OBJECT   as CREATE_DATASET_OBJECT_TRAIN;
         CREATE_DATASET_OBJECT   as CREATE_DATASET_OBJECT_VALIDATION; 
         CREATE_DATASET_OBJECT   as CREATE_DATASET_OBJECT_TEST } from '../modules/create_dataset_object'
include { PREP_INPUT_FILE as PREP_INPUT_FILE_TRAIN; 
          PREP_INPUT_FILE as PREP_INPUT_FILE_VALIDATION; 
          PREP_INPUT_FILE as PREP_INPUT_FILE_TEST } from '../modules/prep_input_file'

include { SPLIT_FILE as SPLIT_TRAIN_BED; 
          SPLIT_FILE as SPLIT_VALIDATION_BED; 
          SPLIT_FILE as SPLIT_TEST_BED } from '../modules/split_file'

include { SUBSAMPLE_BED as SUBSAMPLE_BED_TRAINING;
            SUBSAMPLE_BED as SUBSAMPLE_BED_VALIDATION;
            SUBSAMPLE_BED as SUBSAMPLE_BED_TEST } from '../modules/SUBSAMPLE_BED'


workflow PREPARE_DATA {

    take: 
    genome
    chunk_size
    encode_sheet
    blacklist
    
    main: 

    // Download and preprocess the data
    if (params.fetch_data){
        // Fetch the data
        FETCH_DATA (encode_sheet)
        functional_data = PREPROCESS_DATA(FETCH_DATA.out.data, blacklist)

        // Split dataset into train, test, and validation sets
        // and extract the corresponding fasta files
        SPLIT_DATA(genome, chunk_size)
        genome_chunks_for_training = SPLIT_DATA.out.train
        genome_chunks_for_validation = SPLIT_DATA.out.validation
        genome_chunks_for_testing = SPLIT_DATA.out.test


        if (params.subsample){
            SUBSAMPLE_BED_TRAINING(genome_chunks_for_training, "${params.subsample_size}")
            SUBSAMPLE_BED_VALIDATION(genome_chunks_for_validation, "${params.subsample_size}")
            SUBSAMPLE_BED_TEST(genome_chunks_for_testing, "${params.subsample_size}")
            genome_chunks_for_training = SUBSAMPLE_BED_TRAINING.out.bed
            genome_chunks_for_validation = SUBSAMPLE_BED_VALIDATION.out.bed
            genome_chunks_for_testing = SUBSAMPLE_BED_TEST.out.bed
        }


        train_fasta      = PREP_INPUT_FILE_TRAIN(genome_chunks_for_training, functional_data, genome, "${params.bin_size}", "sum")
        validation_fasta = PREP_INPUT_FILE_VALIDATION(genome_chunks_for_validation, functional_data, genome, "${params.bin_size}", "sum")
        test_fasta       = PREP_INPUT_FILE_TEST(genome_chunks_for_testing, functional_data, genome, "${params.bin_size}", "sum")    
        
        // If too big, split the bed file into smaller files
        SPLIT_TRAIN_BED(train_fasta, "${params.bed_file_max_size}")
        SPLIT_VALIDATION_BED(validation_fasta, "${params.bed_file_max_size}")
        SPLIT_TEST_BED(test_fasta, "${params.bed_file_max_size}")
        
        // Create dataset objects
        train      = CREATE_DATASET_OBJECT_TRAIN(SPLIT_TRAIN_BED.out.split_files.transpose()).unique()
        validation = CREATE_DATASET_OBJECT_VALIDATION(SPLIT_VALIDATION_BED.out.split_files.transpose()).unique()
        test       = CREATE_DATASET_OBJECT_TEST(SPLIT_TEST_BED.out.split_files.transpose()).unique()

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