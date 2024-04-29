
include{ BIG_TO_BEDGRAPH    } from '../modules/BIG_TO_BEDGRAPH'
include{ CLIP_BEDGRAPH      } from '../modules/CLIP_BEDGRAPH'
include{ SUBSET_FOR_TESTING } from '../modules/SUBSET_FOR_TESTING'
include{ CORRECT_BLACKLIST  } from '../modules/CORRECT_BLACKLIST'

workflow PREPROCESS_DATA{

    take: 
    functional_data // [metadata, data]
    

    main:

    // The ones for which file_format is big_wig should be converted to bedgraph
    functional_data_branched = functional_data.branch{
                            bigWig: it[0]["file_format"] == "bigWig"
                        }

    BIG_TO_BEDGRAPH(functional_data_branched.bigWig)
    bedgraph = BIG_TO_BEDGRAPH.out.bedgraph

    // ---------------------------------------------------------
    //   SOFT CLIPPING
    // ---------------------------------------------------------
    if (params.clip){
        CLIP_BEDGRAPH(bedgraph, params.clip_threshold)
        bedgraph = CLIP_BEDGRAPH.out.bedgraph
    }

    emit:
    data = bedgraph

}