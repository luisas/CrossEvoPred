#!/usr/bin/env python
import torch
import argparse
from crossevopred.src.model.trainer import Trainer
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
import importlib

parser = argparse.ArgumentParser(description='Train a model on a given dataset')
parser.add_argument('--training_dataset', nargs='+', help='path to training data')
# training dataset is a list of paths to the training datasets
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--model_name', type=str, help='path to model')
parser.add_argument('--trainer_name', type=str, help='path to trainer')
parser.add_argument('--model_type', type=str, help='name of model to use')
args = parser.parse_args()
print("Arguments parsed")

module = importlib.import_module("crossevopred.src.model."+args.model_type)

# seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Concat the training dataset
# args.training_dataset is a list of paths to the training datasets
# We concatenate them to create a single dataset
trainining_datatset_loader = None

if len(args.training_dataset) == 1:
    trainining_datatset_loader = torch.load(args.training_dataset[0])
else: 
    for dataset in args.training_dataset:
        if trainining_datatset_loader is None:
            trainining_datatset_loader = torch.load(dataset)
        else:
            trainining_datatset_loader = ConcatDataset([trainining_datatset_loader, torch.load(dataset)])
trainining_datatset_loader = DataLoader(trainining_datatset_loader, batch_size=32, shuffle=True)
print("Data loader ready")

# Initialize the model and trainer 
model = getattr(module, args.model_type)()
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
model.save_model( args.model_name )

print("Triner name: ", args.trainer_name)
trainer.save( args.trainer_name )