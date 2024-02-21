#!/usr/bin/env python
import torch
import argparse
from crossevopred.src.model.dummy_model import DummyModel
from bin.crossevopred.src.model.trainer import Trainer
import numpy as np

parser = argparse.ArgumentParser(description='Train a model on a given dataset')
parser.add_argument('--training_dataset', type=str, help='path to training data')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--model_name', type=str, help='path to model')
args = parser.parse_args()
print("Arguments parsed")

# seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


trainining_datatset_loader = torch.load(args.training_dataset)
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

# Train the model
trainer.train( config_file=args.config )

# Save the model 
model.save_model(args.model_name)