import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from .loss import NegativeELBOLoss, MultimodalJSDLoss


class VariationalAutoencoder(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            kl_free_bits=None,
            beta_schedule='constant',
            beta=1.0,
            beta_start=0.0,
            beta_stop=1.0,
            beta_cyclical_total_steps=100,
            beta_cycles=1,
            beta_cycle_ratio_zero=0.5,
            beta_cycle_ratio_increase=0.25,
            loss_reduction='mean'):
        super(VariationalAutoencoder, self).__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = IdentityEncoderDecoder()

        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = IdentityEncoderDecoder()

        self.negative_elbo_loss = NegativeELBOLoss(
            kl_free_bits=kl_free_bits,
            beta_schedule=beta_schedule,
            beta=beta,
            beta_start=beta_start,
            beta_stop=beta_stop,
            beta_cyclical_total_steps=beta_cyclical_total_steps,
            beta_cycles=beta_cycles,
            beta_cycle_ratio_zero=beta_cycle_ratio_zero,
            beta_cycle_ratio_increase=beta_cycle_ratio_increase,
            reduction=loss_reduction)

    def parameter_dicts(self):
        return self.encoder.parameter_dicts() + self.decoder.parameter_dicts()

    def forward(self, encoder_inputs, decoder_inputs, calc_loss=False):
        if encoder_inputs is None:
            encoder_inputs = {}
        encoder_outputs = self.encoder(**encoder_inputs)

        loc = encoder_outputs['loc']
        logscale = encoder_outputs['logscale']

        posterior_q_z_x = distributions.Normal(loc, torch.exp(logscale))

        # Sample using reparameterization trick
        latent_z = posterior_q_z_x.rsample()

        if decoder_inputs is None:
            decoder_inputs = {}
        decoder_inputs.update(encoder_outputs)
        decoder_inputs['latent_z'] = latent_z
        decoder_inputs['calc_loss'] = calc_loss

        decoder_outputs = self.decoder(**decoder_inputs)

        vae_outputs = {
            'posterior_q_z_x': posterior_q_z_x,
            'latent_z': latent_z,
        }

        if calc_loss:
            reconstruction_loss = decoder_outputs['loss']
            vae_outputs['beta'] = self.negative_elbo_loss.beta()

            loss, reconstruction_loss, kl_loss = self.negative_elbo_loss(
                reconstruction_loss, posterior_q_z_x)
            vae_outputs['loss'] = loss
            vae_outputs['reconstruction_loss'] = reconstruction_loss
            vae_outputs['kl_loss'] = kl_loss

        outputs = {
            'encoder_outputs': encoder_outputs,
            'decoder_outputs': decoder_outputs,
            'vae_outputs': vae_outputs,
        }

        return outputs


class MultimodalVariationalAutoencoder(nn.Module):
    def __init__(
            self,
            encoder_decoder_models,
            subsample=False,
            kl_free_bits=None,
            beta_schedule='constant',
            beta=1.0,
            beta_start=0.0,
            beta_stop=1.0,
            beta_cyclical_total_steps=100,
            beta_cycles=1,
            beta_cycle_ratio_zero=0.5,
            beta_cycle_ratio_increase=0.25,
            loss_reduction='mean'):
        super(MultimodalVariationalAutoencoder, self).__init__()
        encoder_decoder_models = [
            (
                encoder if encoder is not None else IdentityEncoderDecoder(),
                decoder if decoder is not None else IdentityEncoderDecoder(),
            ) for encoder, decoder in encoder_decoder_models]
        self.encoder_decoder_models = nn.ModuleList(
            [nn.ModuleList(pair) for pair in encoder_decoder_models])

        # TODO: Implement full subsampling
        self.subsample = subsample
        if self.subsample:
            mvae_subsets = []
            for i in range(len(encoder_decoder_models)):
                encoder_decoder_models_subset = [
                        encoder_decoder_models[j] if i == j else (None, None)
                        for j in range(len(encoder_decoder_models))]

                mvae_subset = MultimodalVariationalAutoencoder(
                    encoder_decoder_models_subset,
                    subsample=False,
                    kl_free_bits=kl_free_bits,
                    beta_schedule=beta_schedule,
                    beta=beta,
                    beta_start=beta_start,
                    beta_stop=beta_stop,
                    beta_cyclical_total_steps=beta_cyclical_total_steps,
                    beta_cycles=beta_cycles,
                    beta_cycle_ratio_zero=beta_cycle_ratio_zero,
                    beta_cycle_ratio_increase=beta_cycle_ratio_increase,
                    loss_reduction=loss_reduction)

                mvae_subsets.append(mvae_subset)
            self.mvae_subsets = mvae_subsets

        self.negative_elbo_loss = NegativeELBOLoss(
            kl_free_bits=kl_free_bits,
            beta_schedule=beta_schedule,
            beta=beta,
            beta_start=beta_start,
            beta_stop=beta_stop,
            beta_cyclical_total_steps=beta_cyclical_total_steps,
            beta_cycles=beta_cycles,
            beta_cycle_ratio_zero=beta_cycle_ratio_zero,
            beta_cycle_ratio_increase=beta_cycle_ratio_increase,
            reduction=loss_reduction)

        self.prior_expert_loc = None
        self.prior_expert_logscale = None
        self.prior_expert_size = None
        self.prior_expert_device = None

    def parameter_dicts(self):
        parameter_dicts = []
        for encoder, decoder in self.encoder_decoder_models:
            parameter_dicts += encoder.parameter_dicts()
            parameter_dicts += decoder.parameter_dicts()
        return parameter_dicts

    def prior_expert(self, size, device=None):
        if self.prior_expert_size != size:
            self.prior_expert_size = size
            self.prior_expert_loc = None
            self.prior_expert_logscale = None

        if self.prior_expert_device != device:
            self.prior_expert_device = device
            self.prior_expert_loc = None
            self.prior_expert_logscale = None

        if self.prior_expert_loc is None:
            self.prior_expert_loc = torch.zeros(
                self.prior_expert_size, device=self.prior_expert_device)

        if self.prior_expert_logscale is None:
            self.prior_expert_logscale = torch.zeros(
                self.prior_expert_size, device=self.prior_expert_device)

        return self.prior_expert_loc, self.prior_expert_logscale

    def product_of_experts(self, locs, logscales, eps=1e-8):
        prior_expert_loc, prior_expert_logscale = self.prior_expert(
            locs[0].shape, locs[0].device)
        locs.append(prior_expert_loc)
        logscales.append(prior_expert_logscale)
        locs = torch.stack(locs, dim=-1)
        logscales = torch.stack(logscales, dim=-1)

        vars = torch.exp(2 * logscales)
        inverse_vars = torch.reciprocal(vars + eps)

        joint_var = torch.reciprocal(torch.sum(inverse_vars, dim=-1))
        joint_loc = torch.sum(locs * inverse_vars, dim=-1) * joint_var
        joint_scale = torch.sqrt(joint_var)

        return joint_loc, joint_scale

    def forward(self, encoder_decoder_inputs, calc_loss=False):
        encoder_decoder_inputs = [
            (
                encoder_inputs if encoder_inputs is not None else {},
                decoder_inputs if decoder_inputs is not None else {},
            ) for encoder_inputs, decoder_inputs in encoder_decoder_inputs]

        encoder_outputs = []
        for i in range(len(encoder_decoder_inputs)):
            encoder = self.encoder_decoder_models[i][0]
            encoder_inputs = encoder_decoder_inputs[i][0]
            encoder_outputs.append(encoder(**encoder_inputs))

        locs = []
        logscales = []
        for i in range(len(encoder_outputs)):
            try:
                loc = encoder_outputs[i]['loc']
                logscale = encoder_outputs[i]['logscale']
            except KeyError:
                continue

            locs.append(loc)
            logscales.append(logscale)

        joint_loc, joint_scale = self.product_of_experts(locs, logscales)
        joint_posterior_q_z_x = distributions.Normal(joint_loc, joint_scale)

        # Sample using reparameterization trick
        latent_z = joint_posterior_q_z_x.rsample()

        decoder_outputs = []
        for i in range(len(encoder_decoder_inputs)):
            decoder = self.encoder_decoder_models[i][1]
            decoder_inputs = encoder_decoder_inputs[i][1]

            decoder_inputs.update(encoder_outputs[i])
            decoder_inputs['latent_z'] = latent_z
            decoder_inputs['calc_loss'] = calc_loss

            decoder_outputs.append(decoder(**decoder_inputs))

        mvae_outputs = {
            'joint_posterior_q_z_x': joint_posterior_q_z_x,
            'latent_z': latent_z,
        }

        if self.subsample:
            mvae_subsets_outputs = []
            for mvae_subset in self.mvae_subsets:
                mvae_subsets_outputs.append(
                    mvae_subset(encoder_decoder_inputs, calc_loss=calc_loss))

        if calc_loss:
            reconstruction_losses = []
            for i in range(len(decoder_outputs)):
                try:
                    reconstruction_loss = decoder_outputs[i]['loss']
                except KeyError:
                    continue

                reconstruction_losses.append(reconstruction_loss)
            reconstruction_loss = torch.sum(torch.stack(
                reconstruction_losses, dim=-1), dim=-1)

            mvae_outputs['beta'] = self.negative_elbo_loss.beta()

            loss, reconstruction_loss, kl_loss = self.negative_elbo_loss(
                reconstruction_loss, joint_posterior_q_z_x)

            mvae_outputs['loss'] = loss
            mvae_outputs['reconstruction_loss'] = reconstruction_loss
            mvae_outputs['kl_loss'] = kl_loss

            if self.subsample:
                for i in range(len(mvae_subsets_outputs)):
                    subset_outputs = mvae_subsets_outputs[i]['mvae_outputs']
                    mvae_outputs['loss'] += subset_outputs['loss']
                    mvae_outputs['reconstruction_loss'] += \
                        subset_outputs['reconstruction_loss']
                    mvae_outputs['kl_loss'] += subset_outputs['kl_loss']

        outputs = {
            'encoder_outputs': encoder_outputs,
            'decoder_outputs': decoder_outputs,
            'mvae_outputs': mvae_outputs,
        }

        if self.subsample:
            outputs['mvae_subsets_outputs'] = mvae_subsets_outputs

        return outputs


class MultimodalVariationalAutoencoderJSD(nn.Module):
    def __init__(
            self,
            encoder_decoder_models,
            kl_free_bits=None,
            beta_schedule='constant',
            beta=1.0,
            beta_start=0.0,
            beta_stop=1.0,
            beta_cyclical_total_steps=100,
            beta_cycles=1,
            beta_cycle_ratio_zero=0.5,
            beta_cycle_ratio_increase=0.25,
            loss_reduction='mean'):
        super(MultimodalVariationalAutoencoderJSD, self).__init__()
        encoder_decoder_models = [
            (
                encoder if encoder is not None else IdentityEncoderDecoder(),
                decoder if decoder is not None else IdentityEncoderDecoder(),
            ) for encoder, decoder in encoder_decoder_models]
        self.encoder_decoder_models = nn.ModuleList(
            [nn.ModuleList(pair) for pair in encoder_decoder_models])

        self.mmjsd_loss = MultimodalJSDLoss(
            kl_free_bits=kl_free_bits,
            beta_schedule=beta_schedule,
            beta=beta,
            beta_start=beta_start,
            beta_stop=beta_stop,
            beta_cyclical_total_steps=beta_cyclical_total_steps,
            beta_cycles=beta_cycles,
            beta_cycle_ratio_zero=beta_cycle_ratio_zero,
            beta_cycle_ratio_increase=beta_cycle_ratio_increase,
            reduction=loss_reduction)

    def parameter_dicts(self):
        parameter_dicts = []
        for encoder, decoder in self.encoder_decoder_models:
            parameter_dicts += encoder.parameter_dicts()
            parameter_dicts += decoder.parameter_dicts()
        return parameter_dicts

    def mixture_of_experts(self, locs, logscales):
        batch_size = locs[0].shape[0]
        num_experts = len(locs)
        # TODO: Accept supplied weights instead of uniform
        k = torch.randint(0, num_experts, size=(batch_size,))

        joint_loc = torch.stack(
            [locs[k[i]][i] for i in range(batch_size)], dim=0)
        joint_logscale = torch.stack(
            [logscales[k[i]][i] for i in range(batch_size)], dim=0)

        return joint_loc, joint_logscale

    def forward(self, encoder_decoder_inputs, calc_loss=False):
        encoder_decoder_inputs = [
            (
                encoder_inputs if encoder_inputs is not None else {},
                decoder_inputs if decoder_inputs is not None else {},
            ) for encoder_inputs, decoder_inputs in encoder_decoder_inputs]

        encoder_outputs = []
        for i in range(len(encoder_decoder_inputs)):
            encoder = self.encoder_decoder_models[i][0]
            encoder_inputs = encoder_decoder_inputs[i][0]
            encoder_outputs.append(encoder(**encoder_inputs))

        locs = []
        logscales = []
        for i in range(len(encoder_outputs)):
            try:
                loc = encoder_outputs[i]['loc']
                logscale = encoder_outputs[i]['logscale']
            except KeyError:
                continue

            locs.append(loc)
            logscales.append(logscale)

        joint_loc, joint_logscale = self.mixture_of_experts(locs, logscales)
        joint_posterior_q_z_x = distributions.Normal(
            joint_loc, torch.exp(joint_logscale))

        # Sample using reparameterization trick
        latent_z = joint_posterior_q_z_x.rsample()

        decoder_outputs = []
        for i in range(len(encoder_decoder_inputs)):
            decoder = self.encoder_decoder_models[i][1]
            decoder_inputs = encoder_decoder_inputs[i][1]

            decoder_inputs.update(encoder_outputs[i])
            decoder_inputs['latent_z'] = latent_z
            decoder_inputs['calc_loss'] = calc_loss

            decoder_outputs.append(decoder(**decoder_inputs))

        mvae_outputs = {
            'joint_posterior_q_z_x': joint_posterior_q_z_x,
            'latent_z': latent_z,
        }

        if calc_loss:
            reconstruction_losses = []
            for i in range(len(decoder_outputs)):
                try:
                    reconstruction_loss = decoder_outputs[i]['loss']
                except KeyError:
                    continue

                reconstruction_losses.append(reconstruction_loss)

            unimodal_posteriors_q_z_x = [
                distributions.Normal(loc, torch.exp(logscale))
                for loc, logscale in zip(locs, logscales)]

            mvae_outputs['beta'] = self.mmjsd_loss.beta()

            loss, reconstruction_loss, mmjsd_loss = self.mmjsd_loss(
                reconstruction_losses, unimodal_posteriors_q_z_x)

            mvae_outputs['loss'] = loss
            mvae_outputs['reconstruction_loss'] = reconstruction_loss
            mvae_outputs['mmjsd_loss'] = mmjsd_loss

        outputs = {
            'encoder_outputs': encoder_outputs,
            'decoder_outputs': decoder_outputs,
            'mvae_outputs': mvae_outputs,
        }

        return outputs


class IdentityEncoderDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(IdentityEncoderDecoder, self).__init__()

    def parameter_dicts(self):
        return []

    def forward(self, **kwargs):
        return kwargs


class FeatureEncoder(BaseModel):
    def __init__(
            self,
            num_features,
            hidden_size,
            latent_size,
            use_embeddings=False,
            dropout_rate=0.1):
        super(FeatureEncoder, self).__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.use_embeddings = use_embeddings
        self.dropout_rate = dropout_rate

        if use_embeddings:
            self.embeddings = nn.Embedding(num_features, hidden_size)
            self.feature_fc = nn.Linear(hidden_size, hidden_size)
        else:
            self.feature_fc = nn.Linear(num_features, hidden_size)

        self.hidden_fc = nn.Linear(hidden_size, hidden_size)
        self.loc_fc = nn.Linear(hidden_size, latent_size, bias=False)
        self.logscale_fc = nn.Linear(hidden_size, latent_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

        self.init_parameters()

    def config(self):
        config = {
            'num_features': self.num_features,
            'hidden_size': self.hidden_size,
            'latent_size': self.latent_size,
            'use_embeddings': self.use_embeddings,
            'dropout_rate': self.dropout_rate,
        }
        return config

    def init_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(
                self.feature_fc.weight, nonlinearity='relu')
            nn.init.constant_(self.feature_fc.bias, 0)

            nn.init.kaiming_uniform_(
                self.hidden_fc.weight, nonlinearity='relu')
            nn.init.constant_(self.hidden_fc.bias, 0)

            nn.init.kaiming_uniform_(
                self.loc_fc.weight, nonlinearity='linear')
            nn.init.kaiming_uniform_(
                self.logscale_fc.weight, nonlinearity='linear')

    def reset_parameters(self):
        if self.use_embeddings:
            self.embeddings.reset_parameters()

        self.feature_fc.reset_parameters()
        self.hidden_fc.reset_parameters()
        self.loc_fc.reset_parameters()
        self.logscale_fc.reset_parameters()

        self.init_parameters()

    def reset_latent(self, new_latent_size=None):
        with torch.no_grad():
            if new_latent_size is not None:
                self.latent_size = new_latent_size
                self.loc_fc = nn.Linear(
                    self.hidden_size, self.latent_size, bias=False)
                self.logscale_fc = nn.Linear(
                    self.hidden_size, self.latent_size, bias=False)

            nn.init.kaiming_uniform_(
                self.loc_fc.weight, nonlinearity='linear')
            nn.init.kaiming_uniform_(
                self.logscale_fc.weight, nonlinearity='linear')

    def forward(self, features, **kwargs):
        if self.use_embeddings:
            features = self.embeddings(features)

        features = F.mish(self.feature_fc(features))
        features = self.dropout(features)
        features = F.mish(self.hidden_fc(features))
        features = self.dropout(features)

        loc = self.loc_fc(features)
        logscale = self.logscale_fc(features)

        encoder_outputs = {
            'loc': loc,
            'logscale': logscale,
        }

        return encoder_outputs


class FeatureDecoder(BaseModel):
    def __init__(
            self,
            num_features,
            hidden_size,
            latent_size,
            dropout_rate=0.1,
            label_smoothing=0.1,
            distribution_type=None):
        super(FeatureDecoder, self).__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing
        self.distribution_type = distribution_type

        self.latent_fc = nn.Linear(latent_size, hidden_size)
        self.hidden_fc = nn.Linear(hidden_size, hidden_size)
        self.feature_fc = nn.Linear(hidden_size, num_features)
        self.dropout = nn.Dropout(dropout_rate)

        self.init_parameters()

    def config(self):
        config = {
            'num_features': self.num_features,
            'hidden_size': self.hidden_size,
            'latent_size': self.latent_size,
            'dropout_rate': self.dropout_rate,
            'label_smoothing': self.label_smoothing,
            'distribution_type': self.distribution_type,
        }
        return config

    def init_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(
                self.latent_fc.weight, nonlinearity='relu')
            nn.init.constant_(self.latent_fc.bias, 0)

            nn.init.kaiming_uniform_(
                self.hidden_fc.weight, nonlinearity='relu')
            nn.init.constant_(self.hidden_fc.bias, 0)

            nn.init.kaiming_uniform_(
                self.feature_fc.weight, nonlinearity='linear')
            nn.init.constant_(self.feature_fc.bias, 0)

    def reset_parameters(self):
        self.latent_fc.reset_parameters()
        self.feature_fc.reset_parameters()
        self.hidden_fc.reset_parameters()

        self.init_parameters()

    def reset_latent(self, new_latent_size=None):
        with torch.no_grad():
            if new_latent_size is not None:
                self.latent_size = new_latent_size
                self.latent_fc = nn.Linear(self.latent_size, self.hidden_size)

            nn.init.kaiming_uniform_(
                self.latent_fc.weight, nonlinearity='relu')
            nn.init.constant_(self.latent_fc.bias, 0)

    def forward(
            self,
            latent_z,
            labels=None,
            calc_loss=False,
            variance=1.0,
            **kwargs):
        if calc_loss and (self.distribution_type is None or labels is None):
            raise ValueError(
                """distribution_type and labels must be provided
                if calc_loss is True""")

        features = F.mish(self.latent_fc(latent_z))
        features = self.dropout(features)
        features = F.mish(self.hidden_fc(features))
        features = self.dropout(features)
        features = self.feature_fc(features)

        decoder_outputs = {
            'features': features,
        }

        if calc_loss:
            if self.distribution_type == 'bernoulli':
                labels = labels.clone()
                labels[labels == 1] = \
                    1.0 - self.label_smoothing
                labels[labels == 0] = \
                    self.label_smoothing

                loss = torch.sum(F.binary_cross_entropy_with_logits(
                    features, labels, reduction='none'), dim=-1)
            elif self.distribution_type == 'categorical':
                loss = F.cross_entropy(features, labels, reduction='none')
            elif self.distribution_type == 'normal':
                loss = torch.sum(
                    F.mse_loss(features, labels, reduction='none'), dim=-1) \
                    / (2.0 * variance)
            else:
                raise ValueError(
                    """distribution_type should be
                    bernoulli, categorical, or normal""")

            decoder_outputs['loss'] = loss

        return decoder_outputs


class LatentClassifier(BaseModel):
    def __init__(
            self,
            latent_size,
            hidden_size,
            num_classes,
            dropout_rate=0.1,
            label_smoothing=0.1):
        super(LatentClassifier, self).__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing

        self.latent_fc = nn.Linear(latent_size, hidden_size)
        self.hidden_fc = nn.Linear(hidden_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        self.init_parameters()

    def config(self):
        config = {
            'latent_size': self.latent_size,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'label_smoothing': self.label_smoothing,
        }
        return config

    def init_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(
                self.latent_fc.weight, nonlinearity='relu')
            nn.init.constant_(self.latent_fc.bias, 0)

            nn.init.kaiming_uniform_(
                self.hidden_fc.weight, nonlinearity='relu')
            nn.init.constant_(self.hidden_fc.bias, 0)

            nn.init.kaiming_uniform_(
                self.output_fc.weight, nonlinearity='linear')
            nn.init.constant_(self.output_fc.bias, 0)

    def reset_parameters(self):
        self.latent_fc.reset_parameters()
        self.hidden_fc.reset_parameters()
        self.output_fc.reset_parameters()

        self.init_parameters()

    def forward(
            self,
            latent_z,
            labels=None,
            calc_loss=False,
            loss_reduction='mean',
            **kwargs):
        if calc_loss and labels is None:
            raise ValueError("labels must be provided if calc_loss is True")

        features = F.mish(self.latent_fc(latent_z))
        features = self.dropout(features)
        features = F.mish(self.hidden_fc(features))
        features = self.dropout(features)
        features = self.output_fc(features)

        outputs = {
            'outputs': features,
        }

        if calc_loss:
            labels = labels.clone()
            labels[labels == 1] = \
                1.0 - self.label_smoothing
            labels[labels == 0] = \
                self.label_smoothing

            loss = F.binary_cross_entropy_with_logits(
                    features, labels, reduction=loss_reduction)
            outputs['loss'] = loss

        return outputs
