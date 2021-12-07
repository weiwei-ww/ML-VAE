import torch
from torch import nn

from speechbrain.nnet.losses import compute_masked_loss

from modules.fc_block import FCBlock


class VanillaVAE(nn.Module):
    def __init__(self, fc_sizes, latent_size):
        super(VanillaVAE, self).__init__()

        self.fc = nn.Sequential(
            FCBlock(fc_sizes),
            nn.LeakyReLU()
        )

        self.mean_fc = nn.Linear(fc_sizes[-1], latent_size)
        self.log_var_fc = nn.Linear(fc_sizes[-1], latent_size)

    def forward(self, feats):  # shape = (B, T, C)
        # B, T = feats.shape[0], feats.shape[1]

        out = self.fc(feats)  # shape = (B, T, C)
        mean = self.mean_fc(out)  # shape = (B, T, C)
        log_var = self.log_var_fc(out)  # shape = (B, T, C)

        # mean = mean.view(B * T, 1, mean.shape[-1])  # shape = (B * T, 1, C)
        # log_var = log_var.view(B * T, 1, log_var.shape[-1])  # shape = (B * T, 1, C)
        sampled_h = self.reparameterize(mean, log_var)  # shape = (B, T, C)

        # loss = self.loss_function(mean, log_var)  # shape = (B, T, C)
        # loss = compute_masked_loss(self.loss_function, mean, log_var, reduction='none')  # shape = (B, T, C)

        return {
            'mean': mean,
            'log_var': log_var,
            'sampled_h': sampled_h
        }

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mean

    def compute_kld_loss(self, mean, log_var):  # shape = (B, T, C)
        kld_loss = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())  # shape = (B, T, C)
        # kld_loss = torch.mean(kld_loss, dim=-1)  # shape = (B * T)
        return kld_loss