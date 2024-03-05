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

        num_tasks = 1
        self.bottleneck_size = 8 
        self.output_length = 24
        
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

        # Linear layer 1 (bottleneck layer)
        self.dense1 = nn.Sequential(
                        nn.Linear(12288,256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Dropout(p=0.3)
                    )
    
        # Rescale to target resolution
        self.dense2 = nn.Sequential(
                        nn.Linear(256,self.output_length * self.bottleneck_size),
                        nn.BatchNorm1d(self.output_length * self.bottleneck_size),
                        nn.ReLU(),
                        Reshape(self.output_length, self.bottleneck_size),
                        nn.Dropout(p=0.1)
                    )

        # Last convolutional layer

        self.last_conv = nn.Sequential(
                            nn.Conv1d(in_channels = 24, out_channels= 256, kernel_size = 7, padding = 7//2),
                            nn.BatchNorm1d(self.bottleneck_size),
                            nn.ReLU(),
                            nn.Dropout(p=0.1)
                    )       


        self.final_linear = nn.Linear(192, self.output_length)

        self.task_conv = nn.Sequential(
                            nn.Conv1d(in_channels = 64, out_channels= num_tasks, kernel_size = 7, padding = 7//2),
                            nn.BatchNorm1d(num_tasks),
                            nn.ReLU(),
                            nn.Dropout(p=0.1)
        )



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        # bottleneck layer
        x = self.dense1(x)
        # now reshape to target resolution
        x = self.dense2(x)
        print(x.shape)
        x = self.last_conv(x)
        print(x.shape)
        x = self.final_linear(x)
        return x


class Reshape(nn.Module):
    def __init__(self, output_length, bottleneck_size):
        super(Reshape, self).__init__()
        self.output_length = output_length
        self.bottleneck_size = bottleneck_size

    def forward(self, x):
        return x.view(-1, self.output_length, self.bottleneck_size)