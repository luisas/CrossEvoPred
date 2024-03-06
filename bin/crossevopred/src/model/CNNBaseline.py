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

        self.num_tasks = 1
        self.bottleneck_size = 8 
        self.output_length = 16
        
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
                        nn.ELU(),
                        nn.MaxPool1d(kernel_size=self.maxpool1_size, stride=self.maxpool1_size),
                        nn.Dropout(p=self.drp1)
                    )
        
        self.conv2 = nn.Sequential(
                        nn.Conv1d(in_channels = self.nfilters_conv1, out_channels= self.nfilters_conv2, kernel_size = self.kernel_size_2, padding= self.kernel_size_2//2),
                        nn.BatchNorm1d(self.nfilters_conv2),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=self.maxpool2_size, stride=self.maxpool2_size),
                        nn.Dropout(p=self.drp2)
                    )
        
        self.conv3 = nn.Sequential(
                        nn.Conv1d(in_channels = self.nfilters_conv2, out_channels= self.nfilters_conv3, kernel_size = self.kernel_size_3, padding= self.kernel_size_3//2),
                        nn.BatchNorm1d(self.nfilters_conv3),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=self.maxpool3_size, stride=self.maxpool3_size),
                        nn.Dropout(p=self.drp3)
                    )
        
        # flatten 
        self.flatten = nn.Flatten()

        # Linear layer 1 (bottleneck layer)
        self.dense1 = nn.Sequential(
                        nn.Linear(8192,256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Dropout(p=0.3)
                    )
    
        # Rescale to target resolution
        self.dense2 = nn.Sequential(
                        nn.Linear(256,self.output_length * self.bottleneck_size),
                        nn.BatchNorm1d(self.output_length * self.bottleneck_size),
                        nn.ReLU(),
                        nn.Dropout(p=0.1)
                    )

        # Last convolutional layer
        self.last_conv = nn.Sequential(
                            nn.Conv1d(in_channels = self.bottleneck_size, out_channels= 256, kernel_size = 7, padding = 7//2),
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            nn.Dropout(p=0.2)
                    )       

        self.task_head = nn.Sequential(
                            nn.Conv1d(in_channels = 256, out_channels= 64, kernel_size = 7, padding = 7//2),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.1)
                    )


        # Final layer 
        self.last_linear = nn.Sequential(
                            nn.Linear(64, 1), 
                            nn.Softplus()
                    )




    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        # bottleneck layer
        x = self.dense1(x)
        # reshape to target resolution
        x = self.dense2(x)
        x = x.view(-1, self.bottleneck_size, self.output_length)
        # final convolutional layer
        x = self.last_conv(x)
        x = self.task_head(x)
        x = x.view(-1, self.output_length, 64)
        x = self.last_linear(x)
        x = x.squeeze()        
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape[0], self.shape[1], self.shape[2])