import torch
import torch.distributions as distributions

from kmvae.mvae.models import MultimodalVariationalAutoencoder

BATCH_SIZE = 4
NUM_MODALITIES = 4
LATENT_SIZE = 32


def test_mvae_product_of_experts():
    locs = [
        torch.rand(BATCH_SIZE, LATENT_SIZE) for _ in range(NUM_MODALITIES)]
    logscales = [
        torch.rand(BATCH_SIZE, LATENT_SIZE) for _ in range(NUM_MODALITIES)]

    # MVAE Joint Posterior
    mvae = MultimodalVariationalAutoencoder([], [])
    joint_loc, joint_scale = mvae.product_of_experts(
        list(locs), list(logscales))
    joint_posterior_q_z_x = distributions.MultivariateNormal(
        joint_loc, scale_tril=torch.diag_embed(joint_scale))

    # Test Joint Posterior
    prior_expert_loc = torch.zeros(BATCH_SIZE, LATENT_SIZE)
    prior_expert_logscale = torch.zeros(BATCH_SIZE, LATENT_SIZE)
    locs.append(prior_expert_loc)
    logscales.append(prior_expert_logscale)
    locs = torch.stack(locs, dim=0)
    logscales = torch.stack(logscales, dim=0)

    covariance_matrices = []
    for i in range(1 + NUM_MODALITIES):
        loc = locs[i]
        logscale = logscales[i]

        posterior_q_z_x = distributions.MultivariateNormal(
            loc, scale_tril=torch.diag_embed(torch.exp(logscale)))
        covariance_matrices.append(posterior_q_z_x.covariance_matrix)
    covariance_matrices = torch.stack(covariance_matrices, dim=0)

    inverse_covariance_matrices = torch.linalg.inv(covariance_matrices)
    joint_covariance_matrix = torch.linalg.inv(
        torch.sum(inverse_covariance_matrices, dim=0))

    locs = torch.unsqueeze(locs, dim=-2)
    joint_loc = torch.matmul(
        torch.sum(torch.matmul(locs, inverse_covariance_matrices), dim=0),
        joint_covariance_matrix)
    joint_loc = torch.squeeze(joint_loc, dim=-2)

    joint_posterior_q_z_x_test = distributions.MultivariateNormal(
        joint_loc, covariance_matrix=joint_covariance_matrix)

    assert torch.allclose(
        joint_posterior_q_z_x.loc,
        joint_posterior_q_z_x_test.loc)

    assert torch.allclose(
        joint_posterior_q_z_x.covariance_matrix,
        joint_posterior_q_z_x_test.covariance_matrix)
