from Bio import SeqIO

def get_fasta_from_coords(chromosome, start, end, fasta_file):
    """"
    Get the fasta sequence from a given chromosome, start and end coordinates   
    """
    for record in SeqIO.parse(fasta_file, 'fasta'):
        if record.id == chromosome:
            return str(record.seq[start:end].upper())

def get_sequences_from_bed(bed_file, fasta_file):
    """
    Get the fasta sequences from a bed file
    """
    sequences = []
    with open(bed_file, 'r') as bed:
        for line in bed:
            chromosome, start, end = line.split()
            sequences.append(get_fasta_from_coords(chromosome, int(start), int(end), fasta_file))
    return sequences


def get_label_from_bedGraph(chromosome, start, end, bedGraph_file, bin_size):
    """
    Get the label from a bedGraph file
    The bin_size is the number of nucleotides that will be binned together
    I need a value for each bin 
    If there is no value for a position, use 0 
    """
    values = []
    with open(bedGraph_file, 'r') as bedGraph:
        for line in bedGraph:
            chrom, start_, end_, value = line.split()
            if chrom == chromosome:
                if int(start_) >= start and int(end_) <= end:
                    values.append(float(value))
    # if values is shorter than the region, add 0s
    if len(values) < (end-start):
        values += [0]*(end-start-len(values))
    # Bin the values
    binned_values = []
    for i in range(0, len(values), bin_size):
        binned_values.append(sum(values[i:i+bin_size])/bin_size)
    return binned_values


def get_labels_from_bedGraph(bed_file, bedGraph_file, bin_size):
    """
    Get the labels from a bedGraph file
    """
    labels = []
    with open(bed_file, 'r') as bed:
        for line in bed:
            # if line is not empty
            if line.strip():
                chromosome, start, end = line.split()
                labels.append(get_label_from_bedGraph(chromosome, int(start), int(end), bedGraph_file, bin_size))
    return labels
