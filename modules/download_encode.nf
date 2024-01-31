process DOWNLOAD_ENCODE{

    input:
    tuple val(meta)

    output:
    tuple val(meta), file("*")

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