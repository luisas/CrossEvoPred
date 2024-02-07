import torch 
from torch.utils.data import Dataset
from ...utils.data_processing_utils import *
from .encoder import *
from ...utils.printing import message 

class ExpressionDataset(Dataset): 
    
    def __init__(self, input_file, encode = True, verbose = True):
        self.verbose = verbose
        
        self.sequences = []
        self.labels = []
        
        with open(input_file, 'r') as file:
            for line in file:
                line = line.strip().split()
                # add sequence to list
                self.sequences.append(line[4].upper())
                # add label to list
                labels = [float(label) for label in line[3].split(',')]
                self.labels.append(labels)

            
        encoder = DNAEncoder()
        if encode:
            self.sequences = encoder.encode_sequences(self.sequences, batch_size=1000)


        self.sequences = torch.tensor(self.sequences).float()
        self.labels = torch.tensor(self.labels).float()


        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx].reshape(1,-1), self.labels[idx]
    
    

