include{ SPLIT_RANDOM } from '../modules/split_random'


workflow SPLIT_DATA {

    take:
    genome
    chunk_size

    main: 
    SPLIT_RANDOM(genome, chunk_size)
}