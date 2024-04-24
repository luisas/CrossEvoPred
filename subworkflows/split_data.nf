include{ SPLIT_RANDOM } from '../modules/split_random'
include{ COBALT } from '../subworkflows/cobalt'
include{ SPLIT_BYCHR } from '../modules/split_bychr'

workflow SPLIT_DATA {

    take:
    genome
    chunk_size

    main: 

    if( params.split = "random"){
        SPLIT_RANDOM(genome, chunk_size)
    }
    else if( params.split = "cobalt"){
        COBALT(genome, chunk_size)
    }
    else if( params.split = "bychr"){
        SPLIT_BYCHR(genome, chunk_size)
    }
    else{
        error "Invalid split method"
    }

    

    emit:
    train = SPLIT_RANDOM.out.train
    validation = SPLIT_RANDOM.out.validation
    test = SPLIT_RANDOM.out.test
}