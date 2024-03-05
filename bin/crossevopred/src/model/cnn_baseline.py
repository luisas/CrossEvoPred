import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from ..data.encoder import *
import torch.nn.init as init

class CNNBaseline(nn.Module):

    def __init__(self):
        super(CNNBaseline, self).__init__()

        # first convolutional layer
        self.nfilters_conv1 = 192
        self.kernel_size_1 = 19
        self.maxpool1_size = 8
        self.drp1 = 0.1

        # second convolutional layer
        self.nfilters_conv2 = 256
        self.kernel_size_2 = 7
        self.maxpool2_size = 4
        self.drp2 = 0.1

        # third convolutional layer
        self.nfilters_conv3 = 512
        self.kernel_size_3 = 7
        self.maxpool3_size = 4
        self.drp3 = 0.2

        # Convolutional layers
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels = 4, out_channels= self.nfilters_conv1, kernel_size = self.kernel_size_1, padding= self.kernel_size_1//2),
                        nn.BatchNorm1d(self.nfilters_conv1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=self.maxpool1_size)
                    )

