
include{ BIG_TO_BEDGRAPH } from '../modules/big_to_bedgraph'
include{ CLIP_BEDGRAPH   } from '../modules/CLIP_BEDGRAPH'
include{ SUBSET_FOR_TESTING } from '../modules/SUBSET_FOR_TESTING'

workflow PREPROCESS_DATA{

    take: 
    functional_data

    main:

    // The ones for which file_format is big_wig should be converted to bedgraph
    functional_data_branched = functional_data.branch{
                            bigWig: it[0]["file_format"] == "bigWig"
                        }

    BIG_TO_BEDGRAPH(functional_data_branched.bigWig)
    bedgraph = BIG_TO_BEDGRAPH.out.bedgraph

    if (params.subset_bed){
        SUBSET_FOR_TESTING(bedgraph)
        bedgraph = SUBSET_FOR_TESTING.out.bedgraph
    }

    bedgraph.view()
    // Add clipping step here
    if (params.clip){
        CLIP_BEDGRAPH(bedgraph, params.clip_threshold)
        bedgraph = CLIP_BEDGRAPH.out.bedgraph
    }

    // if( params.blacklist ){
    //     CORRECT_BLACKLIST(bedgraph, blacklist_regions)
    //     bedgraph = CORRECT_BLACKLIST.out.bedgraph
    // }

    emit:
    data = bedgraph

}