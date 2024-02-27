process FASTA_TO_STOCKHOLM {
    container 'luisas/pytorch_crossevo'    

    input:
    tuple val(meta), path(fasta)

    output:
    tuple val(meta), file("*.sto"), emit: stockholm

    script:
    prefix = task.ext.prefix ?: "${fasta.baseName}"
    """
    #!/usr/bin/env python

    from Bio import SeqIO
    records = SeqIO.parse($fasta, "fasta")
    count = SeqIO.write(records, "${prefix}.sto", "stockholm")
    """
}