
include{ FETCH_DATA               } from '../subworkflows/fetch_data'
include{ SPLIT_DATA               } from '../subworkflows/split_data'
include{ PREPROCESS_DATA          } from '../subworkflows/preprocess_data'
include{ REMOVE_BLACKLIST as REMOVE_BLACKLIST_TRAIN; 
         REMOVE_BLACKLIST as REMOVE_BLACKLIST_VALIDATION; 
         REMOVE_BLACKLIST as REMOVE_BLACKLIST_TEST        } from '../modules/REMOVE_BLACKLIST'

include{ CREATE_DATASET_OBJECT   as CREATE_DATASET_OBJECT_TRAIN;
         CREATE_DATASET_OBJECT   as CREATE_DATASET_OBJECT_VALIDATION; 
         CREATE_DATASET_OBJECT   as CREATE_DATASET_OBJECT_TEST } from '../modules/CREATE_DATASET_OBJECT'

include { PREP_INPUT_FILE as PREP_INPUT_FILE_TRAIN; 
          PREP_INPUT_FILE as PREP_INPUT_FILE_VALIDATION; 
          PREP_INPUT_FILE as PREP_INPUT_FILE_TEST } from '../modules/PREP_INPUT_FILE'

include { CHUNK_FILE as CHUNK_TRAIN_BED; 
          CHUNK_FILE as CHUNK_VALIDATION_BED; 
          CHUNK_FILE as CHUNK_TEST_BED } from '../modules/CHUNK_FILE'

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

        // 
        // Download the data
        //
        FETCH_DATA (encode_sheet)

        // 
        // Preprocess the data
        //
        functional_data = PREPROCESS_DATA(FETCH_DATA.out.data)

        //   
        // Split dataset into train, test, and validation sets
        //
        SPLIT_DATA(genome, chunk_size)
        genome_chunks_for_training = SPLIT_DATA.out.train
        genome_chunks_for_validation = SPLIT_DATA.out.validation
        genome_chunks_for_testing = SPLIT_DATA.out.test


        // 
        // Extract a subsample of the dataset, for quick pipeline testing
        //
        if (params.subsample){
            SUBSAMPLE_BED_TRAINING(genome_chunks_for_training, "${params.subsample_size}")
            SUBSAMPLE_BED_VALIDATION(genome_chunks_for_validation, "${params.subsample_size}")
            SUBSAMPLE_BED_TEST(genome_chunks_for_testing, "${params.subsample_size}")
            genome_chunks_for_training = SUBSAMPLE_BED_TRAINING.out.bed
            genome_chunks_for_validation = SUBSAMPLE_BED_VALIDATION.out.bed
            genome_chunks_for_testing = SUBSAMPLE_BED_TEST.out.bed
        }


        // FILTER BLACKLIST REGIONS
        if (params.blacklist){
            REMOVE_BLACKLIST_TRAIN(genome_chunks_for_training, blacklist)
            REMOVE_BLACKLIST_VALIDATION(genome_chunks_for_validation, blacklist)
            REMOVE_BLACKLIST_TEST(genome_chunks_for_testing, blacklist)
            genome_chunks_for_training = REMOVE_BLACKLIST_TRAIN.out
            genome_chunks_for_validation = REMOVE_BLACKLIST_VALIDATION.out
            genome_chunks_for_testing = REMOVE_BLACKLIST_TEST.out
        }

        //
        // Prepare the bed file with the exact content for the model
        //
        train_fasta      = PREP_INPUT_FILE_TRAIN(genome_chunks_for_training, functional_data, genome, "${params.label_window_size}", "sum", "${params.coverage_threshold}")
        validation_fasta = PREP_INPUT_FILE_VALIDATION(genome_chunks_for_validation, functional_data, genome, "${params.label_window_size}", "sum", "${params.coverage_threshold}")
        test_fasta       = PREP_INPUT_FILE_TEST(genome_chunks_for_testing, functional_data, genome, "${params.label_window_size}", "sum", "${params.coverage_threshold}")    
        
        //
        // If too big, split the bed file into smaller files
        //
        CHUNK_TRAIN_BED(train_fasta, "${params.bed_file_max_size}")
        CHUNK_VALIDATION_BED(validation_fasta, "${params.bed_file_max_size}")
        CHUNK_TEST_BED(test_fasta, "${params.bed_file_max_size}")
        
        //
        // Create dataset objects in pytorch from each of the bed files
        //
        train      = CREATE_DATASET_OBJECT_TRAIN(CHUNK_TRAIN_BED.out.split_files.transpose()).unique()
        validation = CREATE_DATASET_OBJECT_VALIDATION(CHUNK_VALIDATION_BED.out.split_files.transpose()).unique()
        test       = CREATE_DATASET_OBJECT_TEST(CHUNK_TEST_BED.out.split_files.transpose()).unique()

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