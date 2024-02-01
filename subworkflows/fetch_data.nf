
include{ DOWNLOAD_ENCODE } from '../modules/download_encode'

workflow FETCH_DATA {

    take:
    encode_sheet

    main: 
    DOWNLOAD_ENCODE(encode_sheet)
    

    emit: 
    data = DOWNLOAD_ENCODE.out.data
}