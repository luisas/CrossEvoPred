import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from ..eval.losses import PearsonCorrelationLoss
import pandas as pd
from ..data.encoder import *

def swish(x):
    return x * torch.sigmoid(x)


class DummyModel(nn.Module):

    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(4000, 1)

    def forward(self, x):
        x = swish(self.fc1(x))
        return torch.nan_to_num(x)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def evaluate(self, test_dataset_loader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        encoder = DNAEncoder()
        pearson = PearsonCorrelationLoss()
        correlations = []
        with torch.no_grad():
            for sequence, label in test_dataset_loader:
                sequence, label = sequence.to(device), label.to(device)
                output = self(sequence)
                decoded_sequence = encoder.decode_one_hot(sequence.numpy())
                correlation = pearson(output.squeeze(), label.float())
                correlations += [{"sequence": decoded_sequence, "correlation": correlation.item()}]
        return pd.DataFrame(correlations)
        
                
