
#!/usr/bin/env python

import argparse
from torch.utils.data import DataLoader
from .crossevopred.src.model import DummyTrainer
parser = argparse.ArgumentParser(description='Train a model')

# ----------------- Arguments -----------------
# traininig data
parser.add_argument('--training_dataset', type=str, help='path to training data')
# model
parser.add_argument('--model', type=str, help='path to model')
args = parser.parse_args()


train_loader    = DataLoader(args.training_dataset)
validate_loader = DataLoader(args.validation_dataset)

# Initialize the model
trainer = DummyTrainer()

# Train the model
trainer.train(train_loader, validate_loader)

# Save the model 
trainer.save_mode(args.model)