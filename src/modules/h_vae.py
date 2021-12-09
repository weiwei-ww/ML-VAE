import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.fc_block import FCBlock
from modules.vanilla_vae import VanillaVAE
from modules.gmm_vae import GMMVAE

from utils.data_utils import apply_weight


class HierarchicalVAE(nn.Module):
    def __init__(self, fc_sizes, latent_size, num_components):
        super(HierarchicalVAE, self).__init__()

        # VAE for correct pronunciation
        self.vanilla_vae = VanillaVAE(fc_sizes, latent_size)

        # VAE for mispronunciation
        self.gmm_vae = GMMVAE(fc_sizes, latent_size, num_components)

    def forward(
            self,
            feats,  # (B, T, C)
            pi,  # (B, T, 2)
    ):
        vae_in = feats  # (B, T, C)

        # vanilla VAE
        vanilla_out = self.vanilla_vae(vae_in)
        vanilla_mean = vanilla_out['mean']  # (B, T, C)
        vanilla_log_var = vanilla_out['log_var']  # (B, T, C)
        vanilla_h = vanilla_out['sampled_h']  # (B, T, C)
        vanilla_loss = vanilla_out['loss']  # (B, T, C)

        # GMM VAE
        gmm_out = self.gmm_vae(vae_in)
        gmm_mean = gmm_out['mean']  # (B, T, N * C)
        gmm_log_var = gmm_out['log_var']  # (B, T, N * C)
        gmm_h = gmm_out['sampled_h']  # (B, T, N * C)
        gmm_loss = gmm_out['loss']  # (B, T, N * C)

        # apply GMM VAE weights
        gmm_weight = gmm_out['gmm_weight']  # (B, T, N)
        gmm_mean = apply_weight(gmm_mean, gmm_weight)  # (B, T, C)
        gmm_log_var = apply_weight(gmm_log_var, gmm_weight)  # (B, T, C)
        gmm_h = apply_weight(gmm_h, gmm_weight)  # (B, T, C)
        gmm_loss = apply_weight(gmm_loss, gmm_weight)  # (B, T, C)

        # concatenate vanilla and GMM VAE results
        mean = torch.stack([vanilla_mean, gmm_mean], dim=2)  # (B, T, 2, C)
        log_var = torch.stack([vanilla_log_var, gmm_log_var], dim=2)  # (B, T, 2, C)
        h = torch.stack([vanilla_h, gmm_h], dim=2)  # (B, T, 2, C)
        kld_loss = torch.stack([vanilla_loss, gmm_loss], dim=2)  # (B, T, C)

        # compute and apply pi
        mean = apply_weight(mean, pi)  # (B, T, C)
        log_var = apply_weight(log_var, pi)  # (B, T, C)
        h = apply_weight(h, pi)  # (B, T, C)
        kld_loss = apply_weight(kld_loss, pi)  # (B, T, C)

        losses = {
            'vae_kld_loss': kld_loss  # (B, T, C)
        }

        return {
            'gmm_weight': gmm_weight,  # (B, T, N)
            'mean': mean,  # (B, T, C)
            'log_var': log_var,  # (B, T, C)
            'sampled_h': h,  # (B, T, C)
            'losses': losses
        }
