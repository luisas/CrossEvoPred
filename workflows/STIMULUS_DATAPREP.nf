include{ BIG_TO_BEDGRAPH    } from '../modules/BIG_TO_BEDGRAPH'


workflow STIMULUS_DATAPREP {

    take:
    genomes_and_bigwigs // meta, bigwig, genome, genome_index
    

    main:

    bigwigs = genomes_and_bigwigs.map{ meta, bigwig, genome, genome_index ->  [meta,bigwig]}
    BIG_TO_BEDGRAPH(bigwigs)

    // One should be with randomly chunked data

    // One should be with the genes 



}