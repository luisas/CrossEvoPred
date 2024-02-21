#!/usr/bin/env python
import torch
import argparse
from crossevopred.src.model.dummy_model import DummyModel
from bin.crossevopred.src.model.trainer import Trainer
import numpy as np

parser = argparse.ArgumentParser(description='Train a model on a given dataset')
parser.add_argument('--training_dataset', type=str, help='path to training data')
parser.add_argument('--validation_dataset', type=str, help='path to validation data')
parser.add_argument('--tune_config', type=str, help='path to tune config file')
parser.add_argument('--out_config', type=str, help='path to output config file')

args = parser.parse_args()
print("Arguments parsed")

# seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


trainining_datatset_loader = torch.load(args.training_dataset)
validation_datatset_loader = torch.load(args.validation_dataset)
print("Data loaded")

# Initialize the model and trainer 
model   = DummyModel()
model.initialize_weights("xavier_uniform_")
print("Model initialized")

trainer = Trainer(model = model, 
                  training_loader = trainining_datatset_loader,
                  validation_loader = None,
                  test_loader = None,
                  save_training_infos = True,
                  verbose = True)
print("Trainer initialized")

trainer.tune( args.tune_config, best_config_file = args.out_config )
print("Model tuned")