import torch
from torch import nn
import torch.nn.functional as F

from modules.fc_block import FCBlock


class GMMVAE(nn.Module):
    def __init__(self, fc_sizes, latent_size, num_components):
        super(GMMVAE, self).__init__()

        self.fc = nn.Sequential(
            FCBlock(fc_sizes),
            nn.LeakyReLU()
        )

        self.prior_mean_fc = nn.Linear(fc_sizes[-1], latent_size * num_components)
        self.prior_log_var_fc = nn.Linear(fc_sizes[-1], latent_size * num_components)
        self.mean_fc = nn.Linear(fc_sizes[-1], latent_size * num_components)
        self.log_var_fc = nn.Linear(fc_sizes[-1], latent_size * num_components)

        self.gmm_weight_fc = nn.Linear(fc_sizes[-1], num_components)

    def forward(self, feats):
        fc_out = self.fc(feats)  # shape = (B, T, C)
        prior_mean = self.prior_mean_fc(fc_out)  # shape = (B, T, N * C)
        prior_log_var = self.prior_log_var_fc(fc_out)  # shape = (B, T, N * C)
        mean = self.mean_fc(fc_out)  # shape = (B, T, N * C)
        log_var = self.log_var_fc(fc_out)  # shape = (B, T, N * C)
        gmm_weight = self.gmm_weight_fc(fc_out)  # shape = (B, T, N)
        gmm_weight = F.gumbel_softmax(gmm_weight, tau=0.1, hard=True)  # shape = (B, T, N)

        # sampling
        sampled_h = self.reparameterize(mean, log_var)  # shape = (B, T, N * C)

        # compute loss
        loss = self.compute_kld_loss(prior_mean, prior_log_var, mean, log_var)

        ret = {
            'prior_mean': prior_mean,
            'prior_log_var': prior_log_var,
            'mean': mean,  # (B, T, N * C)
            'log_var': log_var,  # (B, T, N * C)
            'sampled_h': sampled_h,  # (B, T, N * C)
            'gmm_weight': gmm_weight,  # (B, T, N)
            'loss': loss  # (B, T, N * C)
        }

        return ret

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mean

        return z

    def compute_kld_loss(self, prior_mean, prior_log_var, mean, log_var):
        # prior_mean, prior_log_var = prediction['prior_mean'], prediction['prior_log_var']
        # mean, log_var = prediction['mean'], prediction['log_var']

        eps = 1e-5
        kld_loss = -0.5 * (
                        1 + log_var - prior_log_var -
                        (log_var.exp() + (mean - prior_mean) ** 2) / (prior_log_var.exp() + eps)
                    )
        return kld_loss
