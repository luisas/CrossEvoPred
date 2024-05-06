include{ FPKM_TO_BED    } from '../modules/FPKM_TO_BED'
include { GET_FASTA } from '../modules/GET_FASTA'
include { BED_TO_STIMULUS } from '../modules/BED_TO_STIMULUS'

workflow STIMULUS_DATAPREP_FPKMS {

    take:
    genomes // meta, genome, genome_index
    fpkms // meta, fpkm

    main:

    fpkms.view()
    FPKM_TO_BED(fpkms)
    beds = FPKM_TO_BED.out.bed

    genomes_and_bed = genomes.map{ meta, genome, index -> [ meta["species"], meta, genome, index ] }
                            .combine(beds.map{ meta, bed -> [ meta["species"], meta, bed ] }, by:0 )
                            .map{ species, meta_genome, genome, index, meta_file, bed -> [["id": species, "species": species], bed, genome, index] }
                            .multiMap{
                                meta, bed, genome, index -> 
                                genome_ch: [ meta, genome, index ]
                                bed_ch: [ meta, bed ]
                            }


    GET_FASTA(genomes_and_bed.bed_ch, genomes_and_bed.genome_ch)

    BED_TO_STIMULUS(GET_FASTA.out.fasta)

}