

process PREPARE_TRAINING_DATA{
    
        input:
        tuple val(meta), file(fasta)
    
        output:
        tuple val(meta), file("*")
    
        script:
        """
        ./prepare_training_data.py -i $fasta -o ${fasta.baseName}.bed
        """
}