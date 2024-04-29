import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--fpkm', help='Input fpkm file', required=True)
    parser.add_argument('-b', '--bed', help='Output bed file', required=True)
    args = parser.parse_args()

    # Read in the fpkm file
    fpkm = pd.read_csv(args.fpkm, sep='\t')
    fpkm["chr"] = fpkm.locus.str.split(":", expand = True)[0]
    fpkm["pos"] = fpkm.locus.str.split(":", expand = True)[1]
    fpkm["start"] = fpkm.pos.str.split("-", expand = True)[0]
    fpkm["end"] = fpkm.pos.str.split("-", expand = True)[1]

    bed = fpkm[["chr", "start", "end", "gene_id", "FPKM"]]
    bed.to_csv(args.bed, index=False, sep = "\t", header = None) 

if __name__ == '__main__':
    main()