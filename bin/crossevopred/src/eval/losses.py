import torch
import torch.nn as nn

class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, y_true, y_pred):
        y_true_mean = torch.mean(y_true)
        y_pred_mean = torch.mean(y_pred)

        numerator = torch.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
        denominator_true = torch.sqrt(torch.sum((y_true - y_true_mean)**2))
        denominator_pred = torch.sqrt(torch.sum((y_pred - y_pred_mean)**2))

        # Avoid division by zero
        eps = 1e-8
        pearson_corr = -numerator / (denominator_true * denominator_pred + eps)

        return pearson_corr.float()