#!/usr/bin/env python
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-b', '--bed', help='Input bed file', required=True)
    parser.add_argument("-s", "--species", help="Input species", required=True)
    parser.add_argument('-o', '--output', help='Output csv file', required=True)
    args = parser.parse_args()

    # Read in the fpkm file
    header = ["sequence:input:dna", "fpkm:label:float", "species:meta:str"]
    bed = pd.read_csv(args.bed, sep='\t')
    bed["species"] = args.species
    # change the column names
    bed.columns = header
    bed.to_csv(args.output, index=False, sep = ",", header = True) 

if __name__ == '__main__':
    main()