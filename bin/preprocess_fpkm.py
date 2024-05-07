#!/usr/bin/env python
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='Read in a fpkm file and generate a processed fpkm file with regions of a given length')
    parser.add_argument('-f', '--fpkm', help='Input fpkm', required=True)
    parser.add_argument('-o', '--output', help='Output processed fpkm file', required=True)
    parser.add_argument('-l', '--length', help='Length of regions', required=True, type=int)
    args = parser.parse_args()

    fpkm = pd.read_csv(args.fpkm, sep = "\t")
    # calculate locus length and save it in length column
    fpkm["start"] = fpkm["locus"].str.split("-").str[0].str.split(":").str[1].astype(int)
    fpkm["end"] = fpkm["locus"].str.split("-").str[1].astype(int)
    fpkm['length'] = fpkm['end'] - fpkm['start']
    
    threshold = args.length 
    fpkm = fpkm[fpkm['length'] > threshold]
    # now for every entry, modify locus end to start + threshold
    fpkm['end'] = fpkm['start'] + threshold
    fpkm['locus'] = fpkm['locus'].str.split(":").str[0] + ":" + fpkm['start'].astype(str) + "-" + fpkm['end'].astype(str)
    fpkm['length'] = fpkm['end'] - fpkm['start']
    fpkm.to_csv(args.output, sep = "\t", index = False)

if __name__ == '__main__':
    main()