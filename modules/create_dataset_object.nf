

process CREATE_DATASET_OBJECT{
    
        container 'luisas/pytorch_crossevo'    
        label "process_medium"

        input:
        tuple val(meta), file(fasta)
        tuple val(meta3), file(bedgraph)
        val(binsize)
    
        output:
        tuple val(meta), file("*.pth"), emit: object
    
        script:
        """
        create_dataset_object.py -f $fasta -bg $bedgraph -bs $binsize -o ${fasta.baseName}.pth
        """
}