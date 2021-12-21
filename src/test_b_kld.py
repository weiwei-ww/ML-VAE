import torch
from torch.distributions.kl import kl_divergence
from torch.distributions import Beta, Kumaraswamy

def compute_kld_loss(alpha, beta):  # shape = (B, T)
    prior_alpha = torch.tensor(1.0)
    prior_beta = torch.tensor(9.0)

    prior_dist = Beta(prior_alpha, prior_beta)

    def beta_function(input_a, input_b):
        ret = torch.exp(torch.lgamma(input_a) + torch.lgamma(input_b) - torch.lgamma(input_a + input_b))
        ret = torch.clamp_min(ret, 1e-5)
        return ret

    kld = 0
    for m in range(1, 100):
        kld = kld + 1 / (m + alpha * beta) * beta_function(m / alpha, beta)

    kld = -(beta - 1) / beta + (prior_beta - 1) * beta * kld
    kld = torch.log(alpha) + torch.log(beta) + torch.log(beta_function(prior_alpha, prior_beta)) + kld

    psi_b = torch.log(beta) - 1 / (2 * beta) - 1 / (12 * beta ** 2)
    kld = kld + (alpha - prior_alpha) / alpha * (-0.57721 - psi_b - 1 / beta)

    # for i in range(kld.shape[0]):
    #     for j in range(kld.shape[1]):
    #         if not torch.isfinite(kld[i, j]):
    #             raise ValueError(f'Invalid value: kld[{i}, {j}], alpha = {alpha[i, j]:.7f}, beta = {beta[i, j]:.7f}')

    return kld


B = 2
T = 5

alpha = torch.rand(B, T)
beta = torch.rand(B, T)
k_dist = Beta(alpha, beta)

prior_alpha = torch.tensor(1.0)
prior_beta = torch.tensor(9.0)
prior_dist = Beta(prior_alpha, prior_beta)

kld = kl_divergence(k_dist, prior_dist)
print(kld)

prior_alpha = torch.full_like(alpha, 1.0)
prior_beta = torch.full_like(beta, 9.0)
prior_dist = Beta(prior_alpha, prior_beta)

kld = kl_divergence(k_dist, prior_dist)
print(kld)


# loss = compute_kld_loss(torch.tensor(alpha), torch.tensor(beta))
#
# prior_alpha = torch.tensor(1.0, requires_grad=True)
# prior_beta = torch.tensor(9.0, requires_grad=True)
#
# prior_dist = Beta(prior_alpha, prior_beta)
# k_dist = Beta(alpha, beta)
#
#
# kld_1 = kl_divergence(prior_dist, k_dist)

# print(loss, kld_0, kld_1)