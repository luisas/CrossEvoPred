process SPLIT_FILE{
    
    input:
    tuple val(meta), path(file)
    val(bed_file_max_size)
    
    output:
    tuple val(meta), path("${prefix}_*"), emit: split_files

    script:
    prefix = task.ext.prefix ?: "${file.baseName}"
    """
    #!/bin/bash
    split -d -l ${bed_file_max_size} ${file} ${prefix}_
    # remove files that have fewer lines than the max size
    # wc -l ${prefix}_* | awk -v max=${bed_file_max_size} '\$1 < max {print \$2}' | xargs rm
    """
}