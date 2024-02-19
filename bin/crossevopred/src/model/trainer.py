import torch.optim as optim
import yaml
from ...utils.printing import message
from .dummy_model import DummyModel
import torch.nn as nn
import pandas as pd
from ray.tune.schedulers import ASHAScheduler
import ray.tune as tune
from ...utils.parsing import create_search_space
import torch
from abc import ABC
import os

class Trainer(ABC):

    def __init__(self, model, training_loader, validation_loader, test_loader, verbose=False) -> None:
        self.verbose = verbose
        # Training infos
        self.training_infos = {}
        # Model 
        self.model = model
        # Datasets
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

    def train(self, config_file, model=None):

        model = DummyModel() if model is None else model

        with open(config_file, "r") as config_file:
            config = yaml.safe_load(config_file)
        
        loss_function = getattr(nn, config['loss_function'])()
        optimizer = getattr(optim, config['optimizer'])(model.parameters(), lr=config['learning_rate'])
        
        # Dictionary with number of eppochs and loss
        training_infos = {"loss": []}
        training_infos["n_epochs"] = config['epochs']
        for epoch in range(config['epochs']):
            message(f"Epoch {epoch+1}/{config['epochs']}", verbose=self.verbose)
            epoch_infos = self._train_epoch(self.training_loader, self.model, optimizer, loss_function)
            training_infos["loss"].append(epoch_infos["loss"])

        training_infos["epoch"] = range(1, len(training_infos["loss"])+1)
        self.training_infos = training_infos

    def tune(self, config, best_config_file = "./ray_tune/best_config.yaml"):

        # Load tuning configuration file
        with open(config, "r") as config_file:
            config = yaml.safe_load(config_file)
        
        # Create scheduler
        scheduler = globals()[config["scheduler"]](**config["scheduler_config"])

        # Create search space
        search_space = create_search_space(config["search_space"])

        # Create tune config
        tune_config = tune.TuneConfig(
            scheduler=scheduler, 
            **config["tune_config"]
        )
        
        # Create tuner
        tuner = tune.Tuner(
            trainable = self.train,
            param_space= search_space, 
            tune_config=tune_config
        )

        # Start tuning
        results = tuner.fit()

        # Save best config
        if not os.path.exists(os.path.dirname(best_config_file)):
            os.makedirs(os.path.dirname(best_config_file))
        with open(best_config_file, "w") as file:
            yaml.dump(results.get_best_result().config, file)

        message(f"Best config saved in {best_config_file}", verbose=self.verbose)
        


    def _train_epoch(self, data_loader, model, optimizer, loss_function):
        # setting device
        # use cuda if available, otherwise use cpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # setting model to train or eval mode
        model.train()

        epoch_infos = {}
        losses = []

        for batch_idx, (sequence, label) in enumerate(data_loader, 0):

            # send data to device
            sequence,label = sequence.to(device), label.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward pass
            output = model(sequence)
            
            # compute loss
            current_loss = loss_function(output.squeeze(), label.float())
            losses.append(current_loss.item())

            # backward and udpate parameters
            current_loss.backward()
            optimizer.step()

        loss = sum(losses) / len(losses)
        epoch_infos["loss"] = (loss)
        message(f"Epoch avg loss: {loss}", verbose=self.verbose)
        return epoch_infos




