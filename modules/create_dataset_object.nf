

process CREATE_DATASET_OBJECT{
    
        container 'luisas/pytorch_crossevo'    
        label "process_lowcpu_highmem"

        input:
        tuple val(meta), file(input_file)
    
        output:
        tuple val(meta), file("*.pth"), emit: object
    
        script:
        """
        create_dataset_object.py -i $input_file -o ${input_file.baseName}.pth 
        """
}