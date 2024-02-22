import torch.optim as optim
import yaml
from ...utils.printing import message
from .dummy_model import DummyModel
import torch.nn as nn
import pandas as pd
from ray.tune.schedulers import ASHAScheduler
import ray.tune as tune
from ray import train
from ...utils.parsing import create_search_space
import torch
from abc import ABC
import os
import json
from ..data.encoder import *

class Trainer(ABC):

    def __init__(self, model, training_loader, validation_loader, test_loader, save_training_infos = True, verbose=False) -> None:
        # General attributes
        self.verbose = verbose
        self.save_training_infos = save_training_infos
        self._tuning = False
        # Training infos
        self.training_infos = {}
        # Model 
        self.model = model
        # Datasets
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

    def train(self, config):
        """
        Train the model using the configuration file

        If _tuning is set to true, the training will be evaluated on the validation set and the loss will be reported to the Ray Tune scheduler
        otherwise, normal training will be performed.

        If self.save_training_infos is set to True, information about the training will be saved in the training_infos attribute
        ( never when tuning )

        Args:
            config (str): Path to the configuration file
    
        """

        # if config is a string, load the configuration file
        if isinstance(config, str):
            with open(config, "r") as config_file:
                config = yaml.safe_load(config_file)

        # Set wether to save training infos (never when tuning)
        if self._tuning:
            save_training_infos = False
        else:
            save_training_infos = self.save_training_infos

        # Set model to training mode
        self.model.train()
        optimizer = getattr(optim, config['optimizer'])(self.model.parameters(), lr=config['learning_rate'])
        
        # Dictionary with number of epochs and loss
        if save_training_infos:
            training_infos = {"loss": []}
            training_infos["n_epochs"] = config['epochs']
            training_infos["epoch"] = list(range(1, config["epochs"]+1))

        # Iterate over epochs
        for epoch in range(config['epochs']):
            message(f"Epoch {epoch+1}/{config['epochs']}", verbose=self.verbose)
            epoch_infos = self._train_epoch(self.training_loader, optimizer, config["loss_function"])
            loss = epoch_infos["loss"]
            message(f"Epoch avg loss: {loss}", verbose=self.verbose)
            if save_training_infos:
                training_infos["loss"].append(loss)

        # Save training infos
        if save_training_infos:   
            self.training_infos = training_infos

        # Evaluate loss on validation set, if tuning
        if self._tuning:
            self.model.eval()
            validation_loss = self._evaluate_loss(self.validation_loader, config["loss_function"])
            train.report(metrics={ "validation_loss": validation_loss })

    def tune(self, config, best_config_file = "./ray_tune/best_config.yaml"):
        """
        Tune the model using the configuration file
        
        Args:
            config (str): Path to the configuration file
            best_config_file (str): Path to the file where the best configuration will be saved
        """
        
        self._tuning = True
        
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
        self._tuning = False

        # Save best config
        if not os.path.exists(os.path.dirname(best_config_file)):
            os.makedirs(os.path.dirname(best_config_file))
        with open(best_config_file, "w") as file:
            yaml.dump(results.get_best_result().config, file)
        
        message(f"Best config saved in {best_config_file}", verbose=self.verbose)
        


    def _train_epoch(self, data_loader, optimizer, loss_function):
        """
        Train the model for one epoch

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader
            optimizer (torch.optim.Optimizer): Optimizer
            loss_function (torch.nn.Module): Loss function
        """
        # setting device
        # use cuda if available, otherwise use cpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        loss_function = getattr(nn, loss_function)()

        epoch_infos = {}
        losses = []

        for batch_idx, (sequence_batch, label_batch) in enumerate(data_loader, 0):

            # send data to device
            sequence_batch,label_batch = sequence_batch.to(device), label_batch.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward pass
            output = self.model(sequence_batch)
            # compute loss
            current_loss = loss_function(output, label_batch)
            losses.append(current_loss.item())

            # backward and udpate parameters
            current_loss.backward()
            optimizer.step()

        loss = sum(losses) / len(losses)
        epoch_infos["loss"] = (loss)
        return epoch_infos


    def _evaluate_loss(self, data_loader, loss_function, all_losses = False):
        """
        Evaluate the loss of the model on a dataset

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader
            loss_function (torch.nn.Module): Loss function
        """

        loss_function = getattr(nn, loss_function)()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        total_loss = 0.0
        num_samples = 0

        if all_losses:
            losses = []
        with torch.no_grad():  # Disable gradient computation
            for batch_idx, (sequence, label) in enumerate(data_loader):
                sequence, label = sequence.to(device), label.to(device)

                output = self.model(sequence)

                loss = loss_function(output.squeeze(), label.float())
                if all_losses:
                    losses += [loss.item()]

                total_loss += loss.item() * sequence.size(0)  
                num_samples += sequence.size(0)

        if all_losses:
            return losses
        else:
            return total_loss / num_samples


