#!/usr/bin/env python
import torch
import argparse
from crossevopred.src.model.dummy_model import DummyModel
from crossevopred.src.model.trainer import Trainer
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader



parser = argparse.ArgumentParser(description='Train a model on a given dataset')
parser.add_argument('--training_dataset', nargs='+', help='path to training data')
# training dataset is a list of paths to the training datasets
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--model_name', type=str, help='path to model')
args = parser.parse_args()
print("Arguments parsed")

# seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Concat the training dataset
# args.training_dataset is a list of paths to the training datasets
# We concatenate them to create a single dataset
trainining_datatset_loader = None
for dataset in args.training_dataset:
    if trainining_datatset_loader is None:
        trainining_datatset_loader = torch.load(dataset)
    else:
        trainining_datatset_loader = ConcatDataset([trainining_datatset_loader, torch.load(dataset)])
trainining_datatset_loader = DataLoader(trainining_datatset_loader, batch_size=32, shuffle=True)
print("Data loader ready")

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
trainer.train( args.config )

# Save the model 
model.save_model(args.model_name)