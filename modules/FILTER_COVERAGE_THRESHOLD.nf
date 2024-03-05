process FILTER_COVERAGE_THRESHOLD {
  tag "Filter the data based on coverage threshold"
  container 'biocontainers/bedtools:v2.27.0_cv3'

  input:
  tuple val(meta), file(bed)
  val (coverage_threshold)
  
  output:
  tuple val(meta), file("${prefix}.bed"), emit: bed
  
  script:
  prefix = task.ext.prefix ?: "${bed.baseName}_filtered"
  """
  #!/bin/bash

  # Filter the bed file based on the coverage threshold
  awk -v threshold=$coverage_threshold '{if ($5 >= threshold) print $0}' ${bed} > ${prefix}.bed
  """
}