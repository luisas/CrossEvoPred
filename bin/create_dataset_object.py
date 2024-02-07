#!/usr/bin/env python

from crossevopred.src.data.dataset import ExpressionDataset
import torch 
import argparse

# read in arguments
parser = argparse.ArgumentParser(description='Create and save dataset object')
parser.add_argument('-i', '--input_file', help='Input bed file with sequences and training labels', required=True)
parser.add_argument('-o', '--output_dataset', help='Output dataset file', required=True)
args = parser.parse_args()

# create dataset object
dataset = ExpressionDataset(args.input_file)

# save dataset object
torch.save(dataset, args.output_dataset)