#!/usr/bin/env python

from crossevopred.src.data.dataset import ExpressionDataset
import torch 
import argparse

# read in arguments
parser = argparse.ArgumentParser(description='Create and save dataset object')
parser.add_argument('-f', '--fasta', help='Input fasta file', required=True)
parser.add_argument('-b', '--bed', help='Input bed file', required=True)
parser.add_argument('-bg', '--bedgraph', help='Input bedgraph file', required=True)
parser.add_argument('-bs', '--binsize', help='Binsize', required=True, type=int)
parser.add_argument('-o', '--output_dataset', help='Output dataset file', required=True)
args = parser.parse_args()

# create dataset object
dataset = ExpressionDataset(args.bed, args.fasta, args.bedgraph, args.binsize)

# save dataset object
torch.save(dataset, args.output_dataset)