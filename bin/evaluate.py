#!/usr/bin/env python
import torch
import argparse
from crossevopred.src.model.dummy_model import DummyModel


parser = argparse.ArgumentParser(description='Test a model on a given dataset')
parser.add_argument('--test_dataset', type=str, help='path to test data')
parser.add_argument('--model', type=str, help='path to model')
parser.add_argument('--output', type=str, help='path to output file')
args = parser.parse_args()
print("Arguments parsed")


model = DummyModel()
model.load_state_dict(torch.load(args.model))
print("Model loaded")
test_dataset_loader = torch.load(args.test_dataset)
print("Test dataset loaded")
evaluation = model.evaluate(test_dataset_loader)
print("Evaluation done")
# Store the evaluation in a file
evaluation.to_csv(args.output, index=False)





