
include{ FETCH_DATA               } from '../subworkflows/fetch_data'
include{ SPLIT_DATA               } from '../subworkflows/split_data'
include{ PREPROCESS_DATA          } from '../subworkflows/preprocess_data'
include{ CREATE_DATASET_OBJECT    } from '../modules/create_dataset_object'

workflow PREPARE_DATA {

    take: 
    genome
    chunk_size
    encode_sheet
    
    main: 

    // Download and preprocess the data
    if (params.fetch_data){
        FETCH_DATA (encode_sheet)
        functional_data = PREPROCESS_DATA(FETCH_DATA.out.data)

        // Split dataset into train, test, and validation sets
        SPLIT_DATA(genome, chunk_size)
        train =  SPLIT_DATA.out.train
        validation = SPLIT_DATA.out.validation
        test = SPLIT_DATA.out.test
    }else{
        // Small test file (this mode is for testing purposes only)
        functional_data = Channel.fromPath("${params.bedgraph_test_file}").map{it -> [[:], it]}
        test_bed_file = Channel.fromPath("${params.test_bed_file}").map{it -> [[:], it]}
        CREATE_DATASET_OBJECT( test_bed_file, genome, functional_data, "${params.bin_size}")
        train =  CREATE_DATASET_OBJECT.out.object
        validation = CREATE_DATASET_OBJECT.out.object
        test = CREATE_DATASET_OBJECT.out.object
        // rename validation and test to avoid conflicts
        // I still want the file, but renamed
        //validation = validation.map{meta, file -> [meta, file.baseName + "_validation.bed"]}
        //test = test.map{meta, file -> [meta, file.baseName + "_test.bed"]}

    } 

    emit: 
    train 
    validation 
    test 


}