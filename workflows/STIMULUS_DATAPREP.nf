include{ BIG_TO_BEDGRAPH    } from '../modules/BIG_TO_BEDGRAPH'


workflow STIMULUS_DATAPREP {

    take:
    genomes_and_bigwigs // meta, bigwig, genome, genome_index
    

    main:

    bigwigs = genomes_and_bigwigs.map{ meta, bigwig, genome, genome_index ->  [meta,bigwig]}

    bigwigs.view()
    BIG_TO_BEDGRAPH(bigwigs)



}