

process CREATE_DATASET_OBJECT{
    
        input:
        tuple val(meta), file(bed)
        tuple val(meta), file(fasta)
        tuple val(meta), file(bedgraph)
        val(binsize)
    
        output:
        tuple val(meta), file("*.pth"), emit: object
    
        script:
        """
        create_dataset_object.py -f $fasta -b $bed -bg $bedgraph -bs $binsize -o ${fasta.baseName}.pth
        """
}