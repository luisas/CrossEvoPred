import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

class DummyModel(nn.Module):

    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        # compute the forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)

