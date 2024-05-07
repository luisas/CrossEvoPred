process PREPROCESS_FPKMS{
    tag "$meta.id"
    label 'process_medium'

    input:
    tuple val(meta), path(fpkm)
    val(length)

    output:
    tuple val(meta), path("*_processed.fpkm"), emit: fpkms

    script:
    def prefix = task.ext.prefix ?: "${fpkm.baseName}_processed.fpkm"
    """
    preprocess_fpkm.py  --fpkm $fpkm \
                     --output ${prefix}_processed.fpkm \
                     --length $length 
    """
}