from collections import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from .models import (ConvE,
                     ConvKB,
                     KBGraphAttentionNetwork,
                     KnowledgeGraphEmbeddings)


class KBGraphAttentionNetworkEncoder(BaseModel):
    def __init__(
            self,
            num_entities,
            num_relations,
            embedding_size,
            num_heads,
            latent_size,
            dropout_rate=0.1,
            pretrained_entity_embeddings=None,
            pretrained_relation_embeddings=None,
            kbgat_state_dict=None):
        super(KBGraphAttentionNetworkEncoder, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.latent_size = latent_size
        self.dropout_rate = dropout_rate

        self.kbgat = KBGraphAttentionNetwork(
            num_entities,
            num_relations,
            embedding_size,
            num_heads,
            dropout_rate=dropout_rate,
            pretrained_entity_embeddings=pretrained_entity_embeddings,
            pretrained_relation_embeddings=pretrained_relation_embeddings)

        if kbgat_state_dict:
            self.kbgat.load_state_dict(kbgat_state_dict, strict=False)

        self.loc_fc = nn.Linear(embedding_size, latent_size, bias=False)
        self.logscale_fc = nn.Linear(embedding_size, latent_size, bias=False)

        self.init_parameters()

    def config(self):
        config = {
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'embedding_size': self.embedding_size,
            'num_heads': self.num_heads,
            'latent_size': self.latent_size,
            'dropout_rate': self.dropout_rate,
        }
        return config

    def init_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(
                self.loc_fc.weight, nonlinearity='linear')
            nn.init.kaiming_uniform_(
                self.logscale_fc.weight, nonlinearity='linear')

    def reset_parameters(self):
        with torch.no_grad():
            self.kbgat.reset_parameters()
            self.loc_fc.reset_parameters()
            self.logscale_fc.reset_parameters()

            self.init_parameters()

    def reset_latent(self, new_latent_size=None):
        with torch.no_grad():
            if new_latent_size is not None:
                self.latent_size = new_latent_size
                self.loc_fc = nn.Linear(
                    self.embedding_size, self.latent_size, bias=False)
                self.logscale_fc = nn.Linear(
                    self.embedding_size, self.latent_size, bias=False)

            self.init_parameters()

    def parameter_dicts(self):
        exclude_weight_decay = [
            'knowledge_graph_embeddings', 'bias', 'LayerNorm.weight']
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

    def forward(
            self,
            entity_ids,
            relation_ids,
            triple_list,
            sparse_triple_adjacency_list_indices,
            context_entity_index,
            **kwargs):
        entity_embeddings, relation_embeddings = self.kbgat(
            entity_ids,
            relation_ids,
            triple_list,
            sparse_triple_adjacency_list_indices)
        context_entities = F.embedding(context_entity_index, entity_embeddings)

        loc = self.loc_fc(context_entities)
        logscale = self.logscale_fc(context_entities)

        encoder_outputs = {
            'entity_embeddings': entity_embeddings,
            'relation_embeddings': relation_embeddings,
            'loc': loc,
            'logscale': logscale,
        }

        return encoder_outputs


class ConvEDecoder(BaseModel):
    def __init__(
            self,
            latent_size,
            num_entities,
            num_relations,
            embedding_size,
            embedding_height,
            num_filters,
            dropout_rate=0.1,
            label_smoothing=0.1,
            knowledge_graph_embeddings_state_dict=None,
            conve_state_dict=None):
        super(ConvEDecoder, self).__init__()
        self.latent_size = latent_size
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_size = embedding_size
        self.embedding_height = embedding_height
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing

        self.latent_fc = nn.Linear(latent_size, embedding_size, bias=False)
        self.knowledge_graph_embeddings = KnowledgeGraphEmbeddings(
            num_entities,
            num_relations,
            embedding_size)
        self.conve = ConvE(
            embedding_size,
            embedding_height,
            num_filters,
            extra_feature_size=embedding_size,
            dropout_rate=dropout_rate)

        if knowledge_graph_embeddings_state_dict:
            self.knowledge_graph_embeddings.load_state_dict(
                knowledge_graph_embeddings_state_dict, strict=False)
        if conve_state_dict:
            self.conve.load_state_dict(conve_state_dict, strict=False)

        self.init_parameters()

    def config(self):
        config = {
            'latent_size': self.latent_size,
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'embedding_size': self.embedding_size,
            'embedding_height': self.embedding_height,
            'num_filters': self.num_filters,
            'dropout_rate': self.dropout_rate,
            'label_smoothing': self.label_smoothing,
        }
        return config

    def init_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(
                self.latent_fc.weight, nonlinearity='linear')

    def reset_parameters(self):
        with torch.no_grad():
            self.latent_fc.reset_parameters()
            self.knowledge_graph_embeddings.reset_parameters()
            self.conve.reset_parameters()

            self.init_parameters()

    def reset_latent(self, new_latent_size=None):
        with torch.no_grad():
            if new_latent_size is not None:
                self.latent_size = new_latent_size
                self.latent_fc = nn.Linear(
                    self.latent_size, self.embedding_size, bias=False)

            self.init_parameters()

    def parameter_dicts(self):
        exclude_weight_decay = [
            'knowledge_graph_embeddings', 'bias', 'LayerNorm.weight']
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

    def forward(
            self,
            latent_z,
            triples,
            triple_labels=None,
            loss_mask=None,
            calc_loss=False,
            **kwargs):
        if calc_loss and triple_labels is None:
            raise ValueError(
                """triple_labels must be provided if calc_loss is True""")

        # Test case with no latent
        # latent_z = torch.zeros_like(latent_z, device=latent_z.device)

        latent_z = self.latent_fc(latent_z)
        latent_z = torch.unsqueeze(latent_z, dim=1)
        latent_z = latent_z.expand(-1, triples.shape[1], -1)

        entity_embeddings, relation_embeddings = \
            self.knowledge_graph_embeddings()

        triple_scores = self.conve(
            triples,
            entity_embeddings,
            relation_embeddings,
            latent_z)

        if loss_mask is not None:
            triple_scores = triple_scores * loss_mask

        if calc_loss:
            triple_labels[triple_labels == 1] = \
                1.0 - self.label_smoothing
            triple_labels[triple_labels == 0] = \
                self.label_smoothing

            loss = F.binary_cross_entropy_with_logits(
                triple_scores, triple_labels, reduction='none')

            if loss_mask is None:
                loss = torch.mean(loss, dim=-1)
            else:
                loss = torch.sum(loss * loss_mask, dim=-1)
                loss = loss / torch.sum(loss_mask, dim=-1)

        decoder_outputs = {
            'triple_scores': triple_scores,
        }

        if calc_loss:
            decoder_outputs['loss'] = loss

        return decoder_outputs


class ConvKBDecoder(nn.Module):
    def __init__(
            self,
            embedding_size,
            num_filters,
            latent_size,
            dropout_rate=0.5,
            convkb_state_dict=None):
        super(ConvKBDecoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        self.latent_size = latent_size
        self.dropout_rate = dropout_rate

        self.convkb = ConvKB(
            embedding_size, num_filters, dropout_rate=dropout_rate)

        if convkb_state_dict:
            self.convkb.load_state_dict(convkb_state_dict)

        self.latent_fc = nn.Linear(latent_size, embedding_size)

    def parameter_dicts(self):
        parameter_dicts = [{'params': self.parameters()}]
        return parameter_dicts

    def forward(
            self,
            triples,
            entity_embeddings,
            relation_embeddings,
            latent_z,
            triple_labels=None,
            calc_loss=False,
            **kwargs):
        if calc_loss and triple_labels is None:
            raise ValueError(
                "triple_labels must be provided if calc_loss is True")

        latent_z = self.latent_fc(latent_z)

        # Handles batched and non-batched inputs
        # TODO: Improve native batching
        if not isinstance(triples, torch.Tensor) \
                and isinstance(triples, abc.Sequence):
            batched = True
        else:
            batched = False
            triples = [triples]
            entity_embeddings = [entity_embeddings]
            relation_embeddings = [relation_embeddings]
            latent_z = [latent_z]
            if calc_loss:
                triple_labels = [triple_labels]

        triple_scores_outputs = []
        if calc_loss:
            loss_outputs = []

        for i in range(len(triples)):
            entity_embeddings[i] = entity_embeddings[i] + latent_z[i]
            triple_scores = self.convkb(
                triples[i], entity_embeddings[i], relation_embeddings[i])
            triple_scores_outputs.append(triple_scores)

            if calc_loss:
                loss = F.binary_cross_entropy_with_logits(
                    triple_scores, triple_labels[i], reduction='sum')
                loss_outputs.append(loss)

        if batched:
            decoder_outputs = {
                'triple_scores': triple_scores_outputs,
            }
            if calc_loss:
                decoder_outputs['loss'] = torch.stack(loss_outputs)
        else:
            decoder_outputs = {
                'triple_scores': triple_scores_outputs[0],
            }
            if calc_loss:
                decoder_outputs['loss'] = loss_outputs[0]

        return decoder_outputs
