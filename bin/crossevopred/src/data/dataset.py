import torch 
from torch.utils.data import Dataset
from ...utils.data_processing_utils import *
from .encoder import *
from ...utils.printing import message 

class ExpressionDataset(Dataset): 
    
    def __init__(self, bed, fasta, bedGraph, bin_size, encode = True, verbose = True):
        self.verbose = verbose
        message("Loading data", verbose=self.verbose)
        encoder = DNAEncoder()
        sequences = get_sequences_from_bed(bed, fasta)
        if encode:
            message("Encoding sequences", verbose=self.verbose)
            self.sequences = encoder.encode_sequences(sequences, batch_size=1000)
        else:
            self.sequences = sequences
        message("Sequences loaded", verbose=self.verbose)
        self.labels = get_labels_from_bedGraph(bed, bedGraph, bin_size)
        message("Labels loaded", verbose=self.verbose)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
    
    

