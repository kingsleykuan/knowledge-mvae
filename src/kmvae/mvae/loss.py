import numpy as np
import torch
import torch.distributions as distributions


# Source: https://github.com/haofuml/cyclical_annealing
def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    # linear schedule
    step = (stop-start)/(period*ratio)

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L


# Source: Line 996 Optimus/code/examples/big_ae/utils.py
def frange_cycle_zero_linear(
        n_iter,
        start=0.0,
        stop=1.0,
        n_cycle=4,
        ratio_increase=0.25,
        ratio_zero=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    # linear schedule
    step = (stop-start)/(period*ratio_increase)

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            if i < period*ratio_zero:
                L[int(i+c*period)] = start
            else:
                L[int(i+c*period)] = v
                v += step
            i += 1
    return L


class NegativeELBOLoss():
    def __init__(
            self,
            kl_free_bits=None,
            beta_schedule='constant',
            beta=1.0,
            beta_start=0.0,
            beta_stop=1.0,
            beta_cyclical_total_steps=100,
            beta_cycles=1,
            beta_cycle_ratio_zero=0.5,
            beta_cycle_ratio_increase=0.25,
            reduction='mean'):
        self.kl_free_bits = kl_free_bits
        self.beta_schedule = beta_schedule
        self._beta = beta
        self.reduction = reduction

        if self.beta_schedule == 'cyclical':
            self.beta_start = beta_start
            self.beta_stop = beta_stop
            self.beta_cyclical_total_steps = beta_cyclical_total_steps
            self.beta_cycles = beta_cycles
            self.beta_cycle_ratio_zero = beta_cycle_ratio_zero
            self.beta_cycle_ratio_increase = beta_cycle_ratio_increase

            self.beta_cyclical_schedule = frange_cycle_zero_linear(
                beta_cyclical_total_steps,
                start=beta_start,
                stop=beta_stop,
                n_cycle=beta_cycles,
                ratio_increase=beta_cycle_ratio_increase,
                ratio_zero=beta_cycle_ratio_zero)

        self.step = 0
        self.device = None
        self.prior_p_z = None

    def beta(self):
        if self.beta_schedule == 'constant':
            return self._beta
        elif self.beta_schedule == 'cyclical':
            if self.step < len(self.beta_cyclical_schedule):
                return self.beta_cyclical_schedule[self.step]
            else:
                return self.beta_stop
        else:
            raise ValueError(
                "beta_schedule should be 'constant' or 'cyclical'")

    def prior(self, device=None):
        if self.device != device:
            self.device = device
            self.prior_p_z = None

        if self.prior_p_z is None:
            self.prior_p_z = distributions.Normal(
                torch.tensor(0, device=self.device),
                torch.tensor(1, device=self.device))

        return self.prior_p_z

    def kl_loss(self, posterior_q_z_x):
        prior_p_z = self.prior(posterior_q_z_x.loc.device)

        kl_divergence = distributions.kl_divergence(posterior_q_z_x, prior_p_z)

        if self.kl_free_bits:
            kl_divergence = (kl_divergence > self.kl_free_bits) * kl_divergence

        kl_divergence = torch.sum(kl_divergence, dim=-1)
        return kl_divergence

    def __call__(self, reconstruction_loss, posterior_q_z_x):
        kl_loss = self.kl_loss(posterior_q_z_x)
        beta = self.beta()

        if self.reduction == 'none':
            reconstruction_loss = reconstruction_loss
            kl_loss = kl_loss
        elif self.reduction == 'mean':
            reconstruction_loss = torch.mean(reconstruction_loss)
            kl_loss = torch.mean(kl_loss)
        elif self.reduction == 'sum':
            reconstruction_loss = torch.sum(reconstruction_loss)
            kl_loss = torch.sum(kl_loss)
        else:
            raise ValueError("reduction should be 'none', 'mean', or 'sum'")

        loss = reconstruction_loss + (beta * kl_loss)
        self.step += 1

        return loss, reconstruction_loss, kl_loss


class MultimodalJSDLoss():
    def __init__(
            self,
            kl_free_bits=None,
            beta_schedule='constant',
            beta=1.0,
            beta_start=0.0,
            beta_stop=1.0,
            beta_cyclical_total_steps=100,
            beta_cycles=1,
            beta_cycle_ratio_zero=0.5,
            beta_cycle_ratio_increase=0.25,
            reduction='mean'):
        self.kl_free_bits = kl_free_bits
        self.beta_schedule = beta_schedule
        self._beta = beta
        self.reduction = reduction

        if self.beta_schedule == 'cyclical':
            self.beta_start = beta_start
            self.beta_stop = beta_stop
            self.beta_cyclical_total_steps = beta_cyclical_total_steps
            self.beta_cycles = beta_cycles
            self.beta_cycle_ratio_zero = beta_cycle_ratio_zero
            self.beta_cycle_ratio_increase = beta_cycle_ratio_increase

            self.beta_cyclical_schedule = frange_cycle_zero_linear(
                beta_cyclical_total_steps,
                start=beta_start,
                stop=beta_stop,
                n_cycle=beta_cycles,
                ratio_increase=beta_cycle_ratio_increase,
                ratio_zero=beta_cycle_ratio_zero)

        self.step = 0

        self.prior_p_z = None
        self.prior_size = None
        self.prior_device = None

    def beta(self):
        if self.beta_schedule == 'constant':
            return self._beta
        elif self.beta_schedule == 'cyclical':
            if self.step < len(self.beta_cyclical_schedule):
                return self.beta_cyclical_schedule[self.step]
            else:
                return self.beta_stop
        else:
            raise ValueError(
                "beta_schedule should be 'constant' or 'cyclical'")

    def prior(self, size, device=None):
        if self.prior_size != size:
            self.prior_size = size
            self.prior_p_z = None

        if self.prior_device != device:
            self.prior_device = device
            self.prior_p_z = None

        if self.prior_p_z is None:
            self.prior_p_z = distributions.Normal(
                torch.zeros(self.prior_size, device=self.prior_device),
                torch.ones(self.prior_size, device=self.prior_device))

        return self.prior_p_z

    def weighted_product_of_experts(self, expert_distributions, eps=1e-8):
        locs = [d.loc for d in expert_distributions]
        vars = [torch.square(d.scale) for d in expert_distributions]

        locs = torch.stack(locs, dim=-1)
        vars = torch.stack(vars, dim=-1)

        inverse_vars = torch.reciprocal(vars + eps)

        # TODO: Accept supplied weights instead of finding mean
        joint_var = torch.reciprocal(torch.mean(inverse_vars, dim=-1))
        joint_loc = torch.mean(locs * inverse_vars, dim=-1) * joint_var
        joint_scale = torch.sqrt(joint_var)

        return distributions.Normal(joint_loc, joint_scale)

    def mmjsd_loss(self, unimodal_posteriors_q_z_x):
        prior_p_z = self.prior(
            unimodal_posteriors_q_z_x[0].loc.shape,
            unimodal_posteriors_q_z_x[0].loc.device)
        expert_distributions = unimodal_posteriors_q_z_x + [prior_p_z]

        dynamic_prior_pf_z_x = self.weighted_product_of_experts(
            expert_distributions)

        # TODO: Accept supplied weights instead of finding mean
        jsd = []
        for unimodal_posterior_q_z_x in unimodal_posteriors_q_z_x:
            kl_divergence = distributions.kl_divergence(
                unimodal_posterior_q_z_x, dynamic_prior_pf_z_x)

            if self.kl_free_bits:
                kl_divergence = \
                    (kl_divergence > self.kl_free_bits) * kl_divergence

            kl_divergence = torch.sum(kl_divergence, dim=-1)
            jsd.append(kl_divergence)
        jsd = torch.mean(torch.stack(jsd, dim=-1), dim=-1)

        return jsd

    def __call__(self, reconstruction_losses, unimodal_posteriors_q_z_x):
        reconstruction_loss = torch.sum(torch.stack(
            reconstruction_losses, dim=-1), dim=-1)
        mmjsd_loss = self.mmjsd_loss(unimodal_posteriors_q_z_x)
        beta = self.beta()

        if self.reduction == 'none':
            reconstruction_loss = reconstruction_loss
            mmjsd_loss = mmjsd_loss
        elif self.reduction == 'mean':
            reconstruction_loss = torch.mean(reconstruction_loss)
            mmjsd_loss = torch.mean(mmjsd_loss)
        elif self.reduction == 'sum':
            reconstruction_loss = torch.sum(reconstruction_loss)
            mmjsd_loss = torch.sum(mmjsd_loss)
        else:
            raise ValueError("reduction should be 'none', 'mean', or 'sum'")

        loss = reconstruction_loss + (beta * mmjsd_loss)
        self.step += 1

        return loss, reconstruction_loss, mmjsd_loss
