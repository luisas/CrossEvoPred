#!/usr/bin/env python

# Read in a fasta file and generate a bed file with random non overlapping regions of a given length
# Usage: python split_random.py -i <input_fasta> -o <output_bed> -l <length> 

import argparse
import random
from Bio import SeqIO

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


def main():
    parser = argparse.ArgumentParser(description='Create fasta file')
    parser.add_argument('-f', '--fasta', help='Input fasta file', required=True)
    parser.add_argument('-bg', '--bedgraph', help='Bedgraph file', required=True)
    parser.add_argument('-o', '--output', help='Output files', required=True, type=int)
    args = parser.parse_args()


    # for each sequence in the fasta file
    # extract the position from header >chr1:100-200
    # extact the bedgraph values for the region
    # write the chr, start, end, sequence, label to the output file
    with open(args.fasta, 'r') as fasta:
        for record in SeqIO.parse(fasta, "fasta"):
            chromosome, pos = record.id.split(":")
            start = pos.split("-")[0]
            end = pos.split("-")[1]
            sequence = str(record.seq)
            labels = get_label_from_bedGraph(chromosome, start, end, args.bedgraph, 5)
            with open(args.output, 'w') as output:
                for i in range(len(sequence)):
                    output.write(f"{chromosome}\t{start+i}\t{start+i+1}\t{sequence[i]}\t{labels[i]}\n")
    


if __name__ == '__main__':
    main()