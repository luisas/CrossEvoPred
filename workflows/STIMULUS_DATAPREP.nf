include{ BIG_TO_BEDGRAPH    } from '../modules/BIG_TO_BEDGRAPH'
include{ PREP_INPUT_FILE    } from '../modules/PREP_INPUT_FILE'

workflow STIMULUS_DATAPREP {

    take:
    genomes_and_bigwigs // meta, bigwig, genome, genome_index
    bed // bed file with the genes or regions to be used 

    main:

    bigwigs = genomes_and_bigwigs.map{ meta, bigwig, genome, genome_index ->  [meta,bigwig] }
    genomes = genomes_and_bigwigs.map{ meta, bigwig, genome, genome_index ->  [meta,genome,genome_index] }

    // Convert bigwigs to bedgraph
    BIG_TO_BEDGRAPH(bigwigs)
    bedgraphs = BIG_TO_BEDGRAPH.out.bedgraph
    
    // Combine bedgraphs and genomes
    ch_bedgraph_and_genome = genomes.map{ meta, genome, index -> [ meta["species"], meta, genome, index ] }
                            .combine(bedgraphs.map{ meta, bedgraph -> [ meta["species"], meta, bedgraph ] }, by:0 )
                            .multiMap{ species, meta_genome, genome, index, meta_file, bedgraph ->
                                    genomes: [meta_genome, genome, index]
                                    bedgraphs: [meta_file, bedgraph]}

    // TODO: here remove blacklist 
    PREP_INPUT_FILE(bed, ch_bedgraph_and_genome.bedgraphs, ch_bedgraph_and_genome.genomes, params.window_length_summary, params.summary_metric, params.clip_value)

    
    



}