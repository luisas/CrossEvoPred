import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from ..data.encoder import *
import torch.nn.init as init
from .model import Model

class CNNBaseline(Model):

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
                        nn.MaxPool1d(kernel_size=self.maxpool1_size),
                        nn.Dropout(p=self.drp1)
                    )
        
        self.conv2 = nn.Sequential(
                        nn.Conv1d(in_channels = self.nfilters_conv1, out_channels= self.nfilters_conv2, kernel_size = self.kernel_size_2, padding= self.kernel_size_2//2),
                        nn.BatchNorm1d(self.nfilters_conv2),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=self.maxpool2_size),
                        nn.Dropout(p=self.drp2)
                    )
        
        self.conv3 = nn.Sequential(
                        nn.Conv1d(in_channels = self.nfilters_conv2, out_channels= self.nfilters_conv3, kernel_size = self.kernel_size_3, padding= self.kernel_size_3//2),
                        nn.BatchNorm1d(self.nfilters_conv3),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=self.maxpool3_size),
                        nn.Dropout(p=self.drp3)
                    )
        
        # flatten 
        self.flatten = nn.Flatten()




    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return x
