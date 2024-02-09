import torch
from abc import ABC, abstractmethod
from ...utils.printing import message

class Trainer(ABC):
    def __init__(self, verbose=False) -> None:
        super().__init__()
        self.verbose = verbose

    def train_epoch(self, data_loader, model, optimizer, loss_function, store_parameters=False):
        return self.run_epoch(data_loader, model, optimizer, loss_function, 'train', store_parameters=store_parameters)

    def run_epoch(self, data_loader, model, optimizer, loss_function, do, store_parameters=False):
        # setting device
        # use cuda if available, otherwise use cpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # setting model to train or eval mode
        model.train()

        losses = []

        for batch_idx, (sequence, label) in enumerate(data_loader, 0):

            # send data to device
            sequence,label = sequence.to(device), label.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward pass
            output = model(sequence)
            
            # compute loss
            current_loss = loss_function(output.squeeze(), label.float())
            losses.append(current_loss.item())

            # backward and udpate parameters
            current_loss.backward()
            optimizer.step()

        loss = sum(losses) / len(losses)
        
        message(f"Epoch avg loss: {loss}", verbose=self.verbose)

    @abstractmethod
    def train():
        """
        main training function
        """
        return NotImplemented



