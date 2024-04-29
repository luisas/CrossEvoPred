
include { FPKM_TO_BED } from '../modules/FPKM_TO_BED.nf'

workflow BERTHELOT_DATAPREP {

    take:
    fpkms  
    genome

    main:
    FPKM_TO_BED(fpkms)







}