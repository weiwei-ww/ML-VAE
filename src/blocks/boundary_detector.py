import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.nnet.losses import compute_masked_loss

from blocks.fc_block import FCBlock


class BoundaryDetector(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, rnn_num_layers, fc_sizes):
        super(BoundaryDetector, self).__init__()

        self.rnn = nn.LSTM(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)

        self.fc_alpha = nn.Sequential(
            FCBlock(fc_sizes),
            nn.Softplus()
        )
        self.fc_beta = nn.Sequential(
            FCBlock(fc_sizes),
            nn.Softplus()
        )

        # self.fc_a = nn.Sequential(
        #     linear.Linear(rnn_dim, fc_ab_dim, 1),
        #     nn.Softplus()
        # )
        # self.fc_b = nn.Sequential(
        #     linear.Linear(rnn_dim, fc_ab_dim, 1),
        #     nn.Softplus()
        # )

    def forward(
            self,
            x,  # shape = (B, T, C)
            feat_lens,  # shape = (B, T, C)
            boundary_seqs  # shape = (B, T)
    ):
        # compute parameters
        rnn_out = self.rnn(x)[0]  # shape = (B, T, C)
        v_alpha = self.fc_alpha(rnn_out)  # shape = (B, T, 1)
        v_alpha = torch.squeeze(v_alpha, dim=-1)  # shape = (B, T)
        v_beta = self.fc_beta(rnn_out)  # shape = (B, T, 1)
        v_beta = torch.squeeze(v_beta, dim=-1)  # shape = (B, T)

        # add eps
        eps = 1e-5
        v_alpha = v_alpha + eps
        v_beta = v_beta + eps

        # compute kld loss
        kld_loss = compute_masked_loss(self.kld_loss_function, v_alpha, v_beta, length=feat_lens)

        # sample M times
        sample_times = 10
        losses = {
            'boundary_bce_loss': 0,
            'boundary_kld_loss': kld_loss
        }
        boundary_v = None
        for _ in range(sample_times):
            # sample u
            uniform_u = torch.rand_like(v_alpha)
            uniform_u = uniform_u * 0.98 + 0.01

            # calculate sampled v
            boundary_v_i = (1 - (uniform_u ** (1 / v_beta))) ** (1 / v_alpha)  # shape = (B, T)
            eps = 1e-5
            boundary_v_i = boundary_v_i * (1 - 2 * eps) + eps

            if boundary_v is None:
                boundary_v = torch.zeros_like(boundary_v_i)
            boundary_v = boundary_v + boundary_v_i

            # compute loss
            for i in range(boundary_v_i.shape[0]):
                for j in range(boundary_v_i.shape[1]):
                    if not 0 < boundary_v_i[i, j] < 1:
                        raise ValueError(f'Invalid values: u = {uniform_u[i, j]}, alpha = {v_alpha[i, j]}, ' +
                                         f'beta = {v_beta[i, j]}, boundary_v = {boundary_v_i[i, j]}')
            loss_fn = functools.partial(F.binary_cross_entropy, reduction='none')
            bce_loss = compute_masked_loss(loss_fn, boundary_v_i, boundary_seqs, length=feat_lens)
            losses['boundary_bce_loss'] = losses['boundary_bce_loss'] + bce_loss

        boundary_v = boundary_v / sample_times
        losses['boundary_bce_loss'] = losses['boundary_bce_loss'] / sample_times

        ret = {
            'boundary_v': boundary_v,
            'losses': losses
        }

        return ret

    def kld_loss_function(self, alpha, beta):  # shape = (B, T)
        prior_alpha = torch.tensor(1.0)
        prior_beta = torch.tensor(9.0)

        def beta_function(input_a, input_b):
            ret = torch.exp(torch.lgamma(input_a) + torch.lgamma(input_b) - torch.lgamma(input_a + input_b))
            ret = torch.clamp_min(ret, 1e-5)
            return ret

        kld = 0
        for m in range(1, 11):
            kld = kld + 1 / (m + alpha * beta) * beta_function(m / alpha, beta)

        kld = -(beta - 1) / beta + (prior_beta - 1) * beta * kld
        kld = torch.log(alpha) + torch.log(beta) + torch.log(beta_function(prior_alpha, prior_beta)) + kld

        psi_b = torch.log(beta) - 1 / (2 * beta) - 1 / (12 * beta ** 2)
        kld = kld + (alpha - prior_alpha) / alpha * (-0.57721 - psi_b - 1 / beta)

        for i in range(kld.shape[0]):
            for j in range(kld.shape[1]):
                if not torch.isfinite(kld[i, j]):
                    raise ValueError(f'Invalid value: kld[{i}, {j}], alpha = {alpha[i, j]:.7f}, beta = {beta[i, j]:.7f}')

        return kld
