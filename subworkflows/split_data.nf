include{ SPLIT_RANDOM } from '../modules/split_random'
include{ COBALT } from '../subworkflows/cobalt'

workflow SPLIT_DATA {

    take:
    genome
    chunk_size

    main: 

    SPLIT_RANDOM(genome, chunk_size)

    COBALT(genome, chunk_size)
    
    

    emit:
    train = SPLIT_RANDOM.out.train
    validation = SPLIT_RANDOM.out.validation
    test = SPLIT_RANDOM.out.test
}