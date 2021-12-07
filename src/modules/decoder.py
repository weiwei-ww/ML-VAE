import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from modules.fc_block import FCBlock

class Decoder(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, rnn_num_layers, rnn_dropout, fc_sizes, loss_type='likelihood'):
        super().__init__()

        self.rnn = nn.LSTM(input_size, rnn_hidden_size, rnn_num_layers,
                           dropout=rnn_dropout, bidirectional=True, batch_first=True)
        self.mean_fc = FCBlock(fc_sizes)
        self.log_var_fc = FCBlock(fc_sizes)

        self.loss_type = loss_type

    def forward(self, sampled_h):  # shape = (B, T, C)
        rnn_out = self.rnn(sampled_h)[0]  # shape = (B, T, C)

        mean = self.mean_fc(rnn_out)  # shape = (B, T, C)
        log_var = self.log_var_fc(rnn_out)  # shape = (B, T, C)

        return {
            'mean': mean,
            'log_var': log_var
        }

    def compute_recon_loss(self, prediction, target):  # shape = (B, T, C)
        mean, log_var = prediction['mean'], prediction['log_var']

        if self.loss_type == 'likelihood':
            eps = 1e-5
            likelihood = -0.5 * (torch.log(2 * torch.tensor(np.pi)) + log_var + (target - mean) ** 2 / (torch.exp(log_var) + eps))
            loss = -likelihood
        elif self.loss_type == 'mse':
            loss = F.mse_loss(target, mean, reduction='none')
        else:
            raise ValueError(f'Invalid loss type: {self.loss_type}')

        return loss
