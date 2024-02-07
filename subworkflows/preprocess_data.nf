
include{ BIG_TO_BEDGRAPH } from '../modules/big_to_bedgraph'

workflow PREPROCESS_DATA{

    take: 
    functional_data

    main:

    // The ones for which file_format is big_wig should be converted to bedgraph
    functional_data_branched = functional_data.branch{
                            bigWig: it[0]["file_format"] == "bigWig"
                        }

    BIG_TO_BEDGRAPH(functional_data_branched.bigWig)

    emit:
    data = BIG_TO_BEDGRAPH.out.bedgraph

}