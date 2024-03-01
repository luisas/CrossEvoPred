#!/usr/bin/env python

# random subsample a bed file 
import argparse
import pandas as pd

def random_subsample(bed_file, output_file, subsample_size):
    bed = pd.read_csv(bed_file, sep="\t", header=None)
    subsample = bed.sample(n=subsample_size)
    subsample.to_csv(output_file, sep="\t", header=False, index=False)

def main():
    parser = argparse.ArgumentParser(description='Random subsample a bed file')
    parser.add_argument('bed_file', type=str, help='path to bed file')
    parser.add_argument('output_file', type=str, help='path to output file')
    parser.add_argument('subsample_size', type=int, help='size of the subsample')
    args = parser.parse_args()

    random_subsample(args.bed_file, args.output_file, args.subsample_size)