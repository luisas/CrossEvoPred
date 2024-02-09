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
        self.sequence_length = 1000
        self.label_size = 7
        self.alphabet_size = 4

        # first convolutional layer
        self.nfilters_conv1 = 50
        self.kernel_size_1 = 19
        self.maxpool1_size = 3
        self.maxpool1_stride = 3

        # linear layers
        self.linear_layer1_size = 1000
        self.dropout_1 = 0.3

        # ----------------------------------------------------------------------
        # First convolutional layer
        # ----------------------------------------------------------------------
        length_after_conv1  = self.sequence_length - self.kernel_size_1 + 1
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels = self.alphabet_size, out_channels= self.nfilters_conv1, kernel_size = self.kernel_size_1),
                        nn.BatchNorm1d(length_after_conv1),
                        nn.ReLU(), 
                        nn.MaxPool1d(kernel_size=self.maxpool1_size, stride=self.maxpool1_stride)
                    )
        conv1_output_size = math.floor((length_after_conv1 - self.maxpool1_size ) / self.maxpool1_stride + 1)

        # ----------------------------------------------------------------------
        # First linear layer
        # ----------------------------------------------------------------------
        self.linear_layer1 = nn.Sequential(nn.Linear(conv1_output_size*self.nfilters_conv1, self.linear_layer1_size),
                                           nn.ReLU(),
                                           nn.Dropout(self.dropout_1)) 
        
        self.linear_layer2 = nn.Linear(self.linear_layer1_size, self.label_size)


    def forward(self, x):
        x = self.conv1(x)
        x = self.linear_layer1(x.reshape(-1))
        x = self.linear_layer2(x)
        return x

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
        
                
