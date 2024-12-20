process DOWNLOAD_ENCODE{

    input:
    tuple val(meta)

    output:
    tuple val(meta), file("*"), emit: data

    script:
    """
    if [[ ${meta.link} == *.gz ]]; then
        wget ${meta.link} 
        gunzip *.gz
    else
        wget ${meta.link}
    fi
    """

}