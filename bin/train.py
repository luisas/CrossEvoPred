#!/usr/bin/env python
import torch
import argparse
from crossevopred.src.model.dummy_model import DummyModel
from bin.crossevopred.src.model.tainer import DummyTrainer


parser = argparse.ArgumentParser(description='Train a model on a given dataset')
parser.add_argument('--training_dataset', type=str, help='path to training data')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--model_name', type=str, help='path to model')
args = parser.parse_args()

print("Arguments parsed")
trainining_datatset_loader = torch.load(args.training_dataset)

print("Data loaded")
# Initialize the model and trainer 
model   = DummyModel()
trainer = DummyTrainer()

print("Model initialized")
# Train the model
trainer.train(trainining_datatset_loader, config_file=args.config, model=model)

# Save the model 
model.save_model(args.model_name)