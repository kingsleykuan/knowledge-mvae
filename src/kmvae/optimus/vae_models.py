import json
from pathlib import Path

import torch
import torch.nn as nn

from ..base_model import BaseModel
from .models import OptimusBert, OptimusGPT2


class OptimusBertEncoder(BaseModel):
    def __init__(
            self,
            pretrained_model_name_or_path='bert-base-cased',
            latent_size=32):
        super(OptimusBertEncoder, self).__init__()

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.latent_size = latent_size

        self.bert = OptimusBert.from_pretrained(
            pretrained_model_name_or_path, latent_size=latent_size)

    def config(self):
        config = {
            'latent_size': self.latent_size,
        }
        return config

    def init_parameters(self):
        pass

    def reset_parameters(self):
        pass

    def reset_latent(self, new_latent_size=None):
        with torch.no_grad():
            if new_latent_size is not None:
                self.latent_size = new_latent_size

                config = self.bert.config
                self.bert.linear = nn.Linear(
                    config.hidden_size,
                    2 * self.latent_size,
                    bias=False)

            nn.init.kaiming_uniform_(
                self.bert.linear.weight, nonlinearity='linear')

    def parameter_dicts(self):
        # Line 279 Optimus/code/examples/big_ae/run_lm_vae_training.py
        exclude_weight_decay = ['bias', 'LayerNorm.weight']
        params = [
            param for name, param in self.named_parameters()
            if not any(exclude in name for exclude in exclude_weight_decay)]
        params_exclude_weight_decay = [
            param for name, param in self.named_parameters()
            if any(exclude in name for exclude in exclude_weight_decay)]

        parameter_dicts = [
            {'params': params},
            {'params': params_exclude_weight_decay, 'weight_decay': 0.0},
        ]
        return parameter_dicts

    def save_pretrained(self, model_path, **config_override):
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        config_path = model_path / 'vae_config.json'

        config = self.config()
        config.update(config_override)
        with config_path.open('w') as file:
            json.dump(config, file, indent=4)

        self.bert.save_pretrained(model_path)

    @classmethod
    def from_pretrained(cls, model_path, **config_override):
        model_path = Path(model_path)
        config_path = model_path / 'vae_config.json'

        with config_path.open() as file:
            config = json.load(file)
        config['pretrained_model_name_or_path'] = model_path
        config.update(config_override)

        model = cls(**config)
        return model

    def forward(self, input_ids, attention_mask=None, **kwargs):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True)

        # Line 44 Optimus/code/examples/big_ae/modules/vae.py
        pooler_output = bert_outputs['pooler_output']
        loc, logvar = torch.chunk(self.bert.linear(pooler_output), 2, -1)
        logscale = logvar / 2

        encoder_outputs = {
            'last_hidden_state': bert_outputs['last_hidden_state'],
            'pooler_output': bert_outputs['pooler_output'],
            'loc': loc,
            'logscale': logscale,
        }

        return encoder_outputs


class OptimusGPT2Decoder(BaseModel):
    def __init__(
            self,
            pretrained_model_name_or_path='gpt2',
            latent_size=32,
            latent_as_gpt_emb=True,
            latent_as_gpt_memory=True,
            pad_token_id=None):
        super(OptimusGPT2Decoder, self).__init__()

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.latent_size = latent_size
        self.latent_as_gpt_emb = latent_as_gpt_emb
        self.latent_as_gpt_memory = latent_as_gpt_memory
        self.pad_token_id = pad_token_id

        self.gpt2 = OptimusGPT2.from_pretrained(
            pretrained_model_name_or_path,
            latent_size=latent_size,
            latent_as_gpt_emb=latent_as_gpt_emb,
            latent_as_gpt_memory=latent_as_gpt_memory)

    def config(self):
        config = {
            'latent_size': self.latent_size,
            'latent_as_gpt_emb': self.latent_as_gpt_emb,
            'latent_as_gpt_memory': self.latent_as_gpt_memory,
            'pad_token_id': self.pad_token_id,
        }
        return config

    def init_parameters(self):
        pass

    def reset_parameters(self):
        pass

    def reset_latent(self, new_latent_size=None):
        with torch.no_grad():
            if new_latent_size is not None:
                self.latent_size = new_latent_size
                self.gpt2.config.latent_size = new_latent_size

                config = self.gpt2.config
                self.gpt2.transformer.linear = nn.Linear(
                    config.latent_size,
                    config.hidden_size * config.n_layer,
                    bias=False)
                self.gpt2.transformer.linear_emb = nn.Linear(
                    config.latent_size, config.hidden_size, bias=False)

            nn.init.kaiming_uniform_(
                self.gpt2.transformer.linear.weight, nonlinearity='linear')
            nn.init.kaiming_uniform_(
                self.gpt2.transformer.linear_emb.weight, nonlinearity='linear')

    def parameter_dicts(self):
        # Line 279 Optimus/code/examples/big_ae/run_lm_vae_training.py
        exclude_weight_decay = ['bias', 'LayerNorm.weight']
        params = [
            param for name, param in self.named_parameters()
            if not any(exclude in name for exclude in exclude_weight_decay)]
        params_exclude_weight_decay = [
            param for name, param in self.named_parameters()
            if any(exclude in name for exclude in exclude_weight_decay)]

        parameter_dicts = [
            {'params': params},
            {'params': params_exclude_weight_decay, 'weight_decay': 0.0},
        ]
        return parameter_dicts

    def save_pretrained(self, model_path, **config_override):
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        config_path = model_path / 'vae_config.json'

        config = self.config()
        config.update(config_override)
        with config_path.open('w') as file:
            json.dump(config, file, indent=4)

        self.gpt2.save_pretrained(model_path)

    @classmethod
    def from_pretrained(cls, model_path, **config_override):
        model_path = Path(model_path)
        config_path = model_path / 'vae_config.json'

        with config_path.open() as file:
            config = json.load(file)
        config['pretrained_model_name_or_path'] = model_path
        config.update(config_override)

        model = cls(**config)
        return model

    def forward(
            self,
            input_ids,
            latent_z,
            calc_loss=False,
            **kwargs):
        if calc_loss:
            labels = input_ids
        else:
            labels = None

        # Line 118 Optimus/code/examples/big_ae/modules/vae.py
        gpt2_outputs = self.gpt2(
            input_ids=input_ids,
            past=latent_z,
            labels=labels,
            label_ignore=self.pad_token_id)

        if calc_loss:
            sequence_len = torch.sum(labels != self.pad_token_id, dim=-1)
            decoder_outputs = {
                'loss': gpt2_outputs[0] / sequence_len,
                'logits': gpt2_outputs[1],
            }
        else:
            decoder_outputs = {'logits': gpt2_outputs[0]}

        return decoder_outputs
