import torch
import torch.nn as nn
import torch.nn.init as init


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def save_model(self, path):
        """
        Save the model

        Args:
            path (str): Path to the file where the model will be saved
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved in {path}")
    
    @staticmethod
    def initialize_weights(self, method):
        """
        Initialize the weights of the model

        Args:
            method (str): Method to initialize the weights
        """
        return NotImplementedError
    
    def initialize_weights(self, initializer_name):
        initializer = getattr(init, initializer_name)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                initializer(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                initializer(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                init.constant_(m.running_mean, 0)
                init.constant_(m.running_var, 1)