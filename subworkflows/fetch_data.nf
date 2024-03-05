
include{ DOWNLOAD_ENCODE } from '../modules/DOWNLOAD_ENCODE'

workflow FETCH_DATA {

    take:
    encode_sheet

    main: 
    DOWNLOAD_ENCODE(encode_sheet)
    

    emit: 
    data = DOWNLOAD_ENCODE.out.data
}