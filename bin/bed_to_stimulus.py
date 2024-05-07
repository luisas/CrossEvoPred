#!/usr/bin/env python
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-b', '--bed', help='Input bed file', required=True)
    parser.add_argument("-s", "--species", help="Input species", required=True)
    parser.add_argument('-o', '--output', help='Output csv file', required=True)
    args = parser.parse_args()

    header = ["fpkm:label:float", "sequence:input:dna", "species:meta:str"]
    bed = pd.read_csv(args.bed, sep='\t')
    bed["species"] = args.species
    bed.columns = header
    bed["sequence_lenght"] = len(bed["sequence:input:dna"][0])
    bed["N_count"] = bed["sequence:input:dna"].str.count("N")
    bed = bed[bed["N_count"] != bed["sequence_lenght"]]
    bed = bed.drop(columns = ["sequence_lenght", "N_count"])
    bed.to_csv(args.output, index=False, sep = ",", header = True) 

if __name__ == '__main__':
    main()