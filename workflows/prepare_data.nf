
include{ FETCH_DATA } from '../subworkflows/fetch_data'

workflow PREPARE_DATA {

    take: 
    encode_sheet
    
    main: 
    FETCH_DATA (encode_sheet)

}