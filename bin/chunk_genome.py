#!/usr/bin/env python

# Read in a fasta file and generate a bed file with random non overlapping regions of a given length
# Usage: python split_random.py -i <input_fasta> -o <output_bed> -l <length> 

import argparse
import random
from Bio import SeqIO

def main():
    parser = argparse.ArgumentParser(description='Read in a fasta file and generate a bed file with random non overlapping regions of a given length')
    parser.add_argument('-i', '--input', help='Input fasta file', required=True)
    parser.add_argument('-o', '--output', help='Output bed file', required=True)
    parser.add_argument('-l', '--length', help='Length of regions', required=True, type=int)
    args = parser.parse_args()

    with open(args.output, 'w') as out:
        for record in SeqIO.parse(args.input, 'fasta'):
            seq_length = len(record.seq)
            print(seq_length)
            start = 0
            end = start + args.length
            while end < seq_length:
                out.write(record.id + '\t' + str(start) + '\t' + str(end) + '\n')
                start += args.length
                end = start + args.length
            out.write(record.id + '\t' + str(start) + '\t' + str(seq_length) + '\n')

if __name__ == '__main__':
    main()