
include{ FETCH_DATA      } from '../subworkflows/fetch_data'
include{ SPLIT_DATA      } from '../subworkflows/split_data'
include{ PREPROCESS_DATA } from '../subworkflows/preprocess_data'

workflow PREPARE_DATA {

    take: 
    genome
    chunk_size
    encode_sheet
    
    main: 

    // Download data 
    FETCH_DATA (encode_sheet)
    functional_data = FETCH_DATA.out.data

    // Prepare data (preprocess and transformations)
    PREPROCESS_DATA(functional_data)

    // Split dataset into train, test, and validation sets
    SPLIT_DATA(genome, chunk_size)

    // Prepare training objects
    //PREPARE_TRAINING_DATA( SPLIT_DATA.train, fasta) 

}