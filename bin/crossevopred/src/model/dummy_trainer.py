from .training_wrapper import Trainer
import torch.optim as optim
import yaml
from ...utils.printing import message
from .dummy_model import DummyModel
from ..eval.losses import PearsonCorrelationLoss

class DummyTrainer(Trainer):

    def __init__(self, verbose=False) -> None:
        super().__init__(verbose=verbose)
        self.verbose = verbose

    def train(self, train_set, config_file, model=None):

        model = DummyModel() if model is None else model
        with open(config_file, "r") as config_file:
            config = yaml.safe_load(config_file)
        
        # Pass a loss function to the model
        loss_function = globals()[config["loss_function"]]()

        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        for epoch in range(config['epochs']):
            message(f"Epoch {epoch+1}/{config['epochs']}", verbose=self.verbose)
            self.train_epoch(train_set, model, optimizer, loss_function)



