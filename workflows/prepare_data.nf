
include{ FETCH_DATA } from '../subworkflows/fetch_data'
include{ SPLIT_DATA } from '../subworkflows/split_data'


workflow PREPARE_DATA {

    take: 
    genome
    chunk_size
    encode_sheet
    
    main: 

    // Download data 
    FETCH_DATA (encode_sheet)

    // Split dataset into train, test, and validation sets
    SPLIT_DATA(genome, chunk_size)


}