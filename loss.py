import torch
from torch import nn
from torch.nn import MSELoss
from torch.nn import L1Loss as MAELoss
from torch.nn import functional as F


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def corr_coeff(output, target):
    input_matrix = torch.cat([output, target], dim=1).T
    corr = torch.corrcoef(input_matrix)
    return corr[0, 1]
