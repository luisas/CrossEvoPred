// include famsa align 
include { FAMSA_ALIGN as FAMSA_ALIGN_COBALT} from '../modules/famsa_align'
include { FASTA_TO_STOCKHOLM } from '../modules/FASTA_TO_STOCKHOLM'
include { CHUNK_GENOME } from '../modules/CHUNK_GENOME'
include { GET_FASTA as GET_FASTA_COBALT } from '../modules/GET_FASTA'
include { COBALT_SPLIT } from '../modules/COBALT_SPLIT'

workflow COBALT{

    take: 
    genome
    chunk_size

    main: 
    // Prepare the chunks 
    CHUNK_GENOME(genome, chunk_size)
    GET_FASTA_COBALT(CHUNK_GENOME.out.bed, genome)

    // align the chunks 
    FAMSA_ALIGN_COBALT(GET_FASTA_COBALT.out.fasta, [[:],[]], false)

    // run cobalt split
    FASTA_TO_STOCKHOLM(FAMSA_ALIGN_COBALT.out.alignment)
    COBALT_SPLIT(FASTA_TO_STOCKHOLM.out.stockholm)
    

    emit: 
    train = COBALT_SPLIT.out.train
    validation = COBALT_SPLIT.out.validation
    test = COBALT_SPLIT.out.test

}