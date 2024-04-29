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



    # According to the chromosome name, split the bed file in 3 bed files

    

    # divide randomly the set of sequences in test, train and validation (80 - 10 - 10 )
    # get a bed file as input and split it in 3 bed files
    train_prop = 0.8
    test_prop = 0.1
    validation_prop = 0.1
    with open(args.output, 'r') as bed_file:
        lines = bed_file.readlines()
        random.shuffle(lines)
        train = lines[:int(len(lines) * train_prop)]
        test = lines[int(len(lines) * train_prop):int(len(lines) * (train_prop+test_prop))]
        validation = lines[int(len(lines) * (train_prop+test_prop)):]
        # Save files 
        train_name = args.output.split('.')[0] + '_train.bed'
        test_name = args.output.split('.')[0] + '_test.bed'
        validation_name = args.output.split('.')[0] + '_validation.bed'
        with open(train_name, 'w') as train_file:
            train_file.writelines(train)
        with open(test_name, 'w') as test_file:
            test_file.writelines(test)
        with open(validation_name, 'w') as validation_file:
            validation_file.writelines(validation)

if __name__ == '__main__':
    main()
