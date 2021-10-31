import math

import torch
import torch.linalg as linalg
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeGraphEmbeddings(nn.Module):
    def __init__(
            self,
            num_entities,
            num_relations,
            embedding_size,
            pretrained_entity_embeddings=None,
            pretrained_relation_embeddings=None):
        super(KnowledgeGraphEmbeddings, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_size = embedding_size

        self.entity_embeddings = nn.Embedding(num_entities, embedding_size)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_size)

        self.init_parameters()

        if pretrained_entity_embeddings is not None:
            self.entity_embeddings = nn.Embedding.from_pretrained(
                pretrained_entity_embeddings, freeze=False)

        if pretrained_relation_embeddings is not None:
            self.relation_embeddings = nn.Embedding.from_pretrained(
                pretrained_relation_embeddings, freeze=False)

    def init_parameters(self):
        with torch.no_grad():
            nn.init.uniform_(
                self.entity_embeddings.weight,
                -math.sqrt(6 / self.num_entities),
                math.sqrt(6 / self.num_entities))

            nn.init.uniform_(
                self.relation_embeddings.weight,
                -math.sqrt(6 / self.num_relations),
                math.sqrt(6 / self.num_relations))

            F.normalize(
                self.relation_embeddings.weight,
                p=2, dim=-1,
                out=self.relation_embeddings.weight)

    def reset_parameters(self):
        with torch.no_grad():
            self.init_parameters()

    def parameter_dicts(self):
        parameter_dicts = [{'params': self.parameters(), 'weight_decay': 0}]
        return parameter_dicts

    def forward(self):
        return self.entity_embeddings.weight, self.relation_embeddings.weight


class TransE(nn.Module):
    def __init__(self, norm=1):
        super(TransE, self).__init__()
        self.norm = norm

    def forward(self, triples, entity_embeddings, relation_embeddings):
        with torch.no_grad():
            head_entities = triples[..., 0]
            tail_entities = triples[..., 1]
            relations = triples[..., 2]

        head_features = F.embedding(head_entities, entity_embeddings)
        tail_features = F.embedding(tail_entities, entity_embeddings)
        relation_features = F.embedding(relations, relation_embeddings)
        del head_entities, tail_entities, relations

        head_features = F.normalize(head_features, p=2, dim=-1)
        tail_features = F.normalize(tail_features, p=2, dim=-1)

        difference = head_features + relation_features - tail_features
        return linalg.vector_norm(difference, ord=self.norm, dim=-1)


class ConvKB(nn.Module):
    def __init__(self, embedding_size, num_filters, dropout_rate=0.5):
        super(ConvKB, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

        self.conv = nn.Conv2d(1, num_filters, (1, 3))
        self.dot_product_weight = nn.Parameter(
            torch.empty(embedding_size * num_filters, 1))
        self.dropout = nn.Dropout(dropout_rate)

        self.init_parameters()

    def init_parameters(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.dot_product_weight)

    def reset_parameters(self):
        with torch.no_grad():
            self.init_parameters()
            self.conv.reset_parameters()

    def forward(self, triples, entity_embeddings, relation_embeddings):
        batch_shape = triples.shape[:-1]

        with torch.no_grad():
            triples = torch.reshape(triples, (-1, 3))
            head_entities = triples[:, 0]
            tail_entities = triples[:, 1]
            relations = triples[:, 2]

        head_features = F.embedding(head_entities, entity_embeddings)
        tail_features = F.embedding(tail_entities, entity_embeddings)
        relation_features = F.embedding(relations, relation_embeddings)
        del head_entities, tail_entities, relations

        triple_features = torch.stack(
            (head_features, relation_features, tail_features), dim=-1)
        triple_features = torch.unsqueeze(triple_features, 1)
        del head_features, relation_features, tail_features

        triple_features = self.conv(triple_features)
        triple_features = F.relu(triple_features)

        triple_features = torch.reshape(
            triple_features, (-1, self.num_filters * self.embedding_size))
        triple_features = self.dropout(triple_features)

        score = torch.matmul(triple_features, self.dot_product_weight)
        score = torch.squeeze(score, -1)
        score = torch.reshape(score, batch_shape)

        return score


class ConvE(nn.Module):
    def __init__(
            self,
            embedding_size,
            embedding_height,
            num_filters,
            extra_feature_size=0,
            dropout_rate=0.1):
        super(ConvE, self).__init__()
        self.embedding_size = embedding_size
        self.embedding_height = embedding_height
        self.embedding_width = int(embedding_size / embedding_height)
        self.num_filters = num_filters
        self.extra_feature_size = extra_feature_size
        self.dropout_rate = dropout_rate

        self.hidden_size = (
            ((self.embedding_height * 2) - 2)
            * (self.embedding_width - 2)
            * self.num_filters)

        if self.extra_feature_size > 0:
            self.head_extra_fc = nn.Linear(
                self.embedding_size + self.extra_feature_size,
                self.embedding_size)
            self.tail_extra_fc = nn.Linear(
                self.embedding_size + self.extra_feature_size,
                self.embedding_size)
            self.relation_extra_fc = nn.Linear(
                self.embedding_size + self.extra_feature_size,
                self.embedding_size)

        self.input_layer_norm = nn.LayerNorm(
            (1, self.embedding_height * 2, self.embedding_width))
        self.conv = nn.Conv2d(1, num_filters, (3, 3), bias=False)
        self.conv_layer_norm = nn.LayerNorm((
            num_filters,
            (self.embedding_height * 2) - 2,
            self.embedding_width - 2))
        self.fc = nn.Linear(self.hidden_size, embedding_size, bias=False)
        self.fc_layer_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.init_parameters()

    def init_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(
                self.conv.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(
                self.fc.weight, nonlinearity='relu')

            if self.extra_feature_size > 0:
                nn.init.kaiming_uniform_(
                    self.head_extra_fc.weight, nonlinearity='linear')
                nn.init.constant_(self.head_extra_fc.bias, 0)

                nn.init.kaiming_uniform_(
                    self.tail_extra_fc.weight, nonlinearity='linear')
                nn.init.constant_(self.tail_extra_fc.bias, 0)

                nn.init.kaiming_uniform_(
                    self.relation_extra_fc.weight, nonlinearity='linear')
                nn.init.constant_(self.relation_extra_fc.bias, 0)

    def reset_parameters(self):
        with torch.no_grad():
            self.input_layer_norm.reset_parameters()
            self.conv.reset_parameters()
            self.conv_layer_norm.reset_parameters()
            self.fc.reset_parameters()
            self.fc_layer_norm.reset_parameters()

            if self.extra_feature_size > 0:
                self.head_extra_fc.reset_parameters()
                self.tail_extra_fc.reset_parameters()
                self.relation_extra_fc.reset_parameters()

            self.init_parameters()

    def parameter_dicts(self):
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

    def forward(
            self,
            triples,
            entity_embeddings,
            relation_embeddings,
            extra_features=None):
        batch_shape = triples.shape[:-1]

        with torch.no_grad():
            triples = torch.reshape(triples, (-1, 3))
            head_entities = triples[:, 0]
            tail_entities = triples[:, 1]
            relations = triples[:, 2]

        head_features = F.embedding(head_entities, entity_embeddings)
        tail_features = F.embedding(tail_entities, entity_embeddings)
        relation_features = F.embedding(relations, relation_embeddings)
        del head_entities, tail_entities, relations

        if self.extra_feature_size > 0 and extra_features is not None:
            extra_features = torch.reshape(
                extra_features, (-1, self.embedding_size))
            head_features = self.head_extra_fc(torch.cat(
                (head_features, extra_features), dim=-1))
            tail_features = self.tail_extra_fc(torch.cat(
                (tail_features, extra_features), dim=-1))
            relation_features = self.relation_extra_fc(torch.cat(
                (relation_features, extra_features), dim=-1))

        head_features = torch.reshape(
            head_features,
            (-1, 1, self.embedding_height, self.embedding_width))

        relation_features = torch.reshape(
            relation_features,
            (-1, 1, self.embedding_height, self.embedding_width))

        features = torch.cat((head_features, relation_features), dim=2)
        del head_features, relation_features

        features = self.input_layer_norm(features)
        features = self.dropout(features)

        features = self.conv(features)
        features = self.conv_layer_norm(features)
        features = F.mish(features)
        features = self.dropout(features)

        features = torch.reshape(features, (-1, self.hidden_size))
        features = self.fc(features)
        features = self.dropout(features)
        features = self.fc_layer_norm(features)
        features = F.mish(features)

        features = torch.reshape(
            features, (-1, 1, self.embedding_size))
        tail_features = torch.reshape(
            tail_features, (-1, self.embedding_size, 1))

        scores = torch.matmul(features, tail_features)
        scores = torch.reshape(scores, batch_shape)
        return scores


class KBGraphAttentionNetwork(nn.Module):
    def __init__(
            self,
            num_entities,
            num_relations,
            embedding_size,
            num_heads,
            dropout_rate=0.1,
            pretrained_entity_embeddings=None,
            pretrained_relation_embeddings=None):
        super(KBGraphAttentionNetwork, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.knowledge_graph_embeddings = KnowledgeGraphEmbeddings(
            num_entities,
            num_relations,
            embedding_size,
            pretrained_entity_embeddings=pretrained_entity_embeddings,
            pretrained_relation_embeddings=pretrained_relation_embeddings)

        self.layer_1 = KBGraphAttentionalLayer(
            embedding_size,
            embedding_size,
            embedding_size,
            num_heads,
            concatenate=True,
            dropout_rate=dropout_rate)
        self.entity_layer_norm_1 = nn.LayerNorm(embedding_size)
        self.relation_layer_norm_1 = nn.LayerNorm(embedding_size)

        self.layer_2 = KBGraphAttentionalLayer(
            embedding_size,
            embedding_size,
            embedding_size,
            num_heads,
            concatenate=False,
            dropout_rate=dropout_rate)
        self.entity_layer_norm_2 = nn.LayerNorm(embedding_size)
        self.relation_layer_norm_2 = nn.LayerNorm(embedding_size)

        # self.entity_weight = nn.Parameter(
        #     torch.empty(embedding_size, embedding_size))

        self.entity_fc = nn.Linear(embedding_size, embedding_size, bias=True)
        self.relation_fc = nn.Linear(embedding_size, embedding_size, bias=True)

        self.init_parameters()

    def init_parameters(self):
        with torch.no_grad():
            # nn.init.xavier_uniform_(self.entity_weight)
            nn.init.kaiming_uniform_(
                self.entity_fc.weight, nonlinearity='linear')
            nn.init.constant_(self.entity_fc.bias, 0)

            nn.init.kaiming_uniform_(
                self.relation_fc.weight, nonlinearity='linear')
            nn.init.constant_(self.relation_fc.bias, 0)

    def reset_parameters(self):
        with torch.no_grad():
            self.knowledge_graph_embeddings.reset_parameters()

            self.layer_1.reset_parameters()
            self.entity_layer_norm_1.reset_parameters()
            self.relation_layer_norm_1.reset_parameters()

            self.layer_2.reset_parameters()
            self.entity_layer_norm_2.reset_parameters()
            self.relation_layer_norm_2.reset_parameters()

            self.entity_fc.reset_parameters()
            self.relation_fc.reset_parameters()

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
            sparse_triple_adjacency_list_indices):
        # TODO: auxilary relations
        entity_embeddings, relation_embeddings = \
            self.knowledge_graph_embeddings()
        entity_features = F.embedding(entity_ids, entity_embeddings)
        relation_features = F.embedding(relation_ids, relation_embeddings)
        del entity_embeddings, relation_embeddings

        # original_entity_features = entity_features

        entity_features, relation_features = self.layer_1(
            entity_features,
            relation_features,
            triple_list,
            sparse_triple_adjacency_list_indices)
        entity_features = self.entity_layer_norm_1(entity_features)
        relation_features = self.relation_layer_norm_1(relation_features)
        entity_features = F.mish(entity_features)
        relation_features = F.mish(relation_features)

        entity_features, relation_features = self.layer_2(
            entity_features,
            relation_features,
            triple_list,
            sparse_triple_adjacency_list_indices)
        entity_features = self.entity_layer_norm_2(entity_features)
        relation_features = self.relation_layer_norm_2(relation_features)
        entity_features = F.mish(entity_features)
        relation_features = F.mish(relation_features)

        # entity_features = entity_features + torch.matmul(
        #     original_entity_features, self.entity_weight)

        entity_features = self.entity_fc(entity_features)
        relation_features = self.relation_fc(relation_features)

        return entity_features, relation_features


class KBGraphAttentionalLayer(nn.Module):
    def __init__(
            self,
            entity_embedding_size_in,
            entity_embedding_size_out,
            relation_embedding_size,
            num_heads,
            concatenate=True,
            dropout_rate=0.1):
        super(KBGraphAttentionalLayer, self).__init__()

        if concatenate:
            entity_embedding_size_out = \
                int(entity_embedding_size_out / num_heads)

        self.entity_embedding_size_in = entity_embedding_size_in
        self.entity_embedding_size_out = entity_embedding_size_out
        self.relation_embedding_size = relation_embedding_size
        self.num_heads = num_heads
        self.concatenate = concatenate
        self.dropout_rate = dropout_rate

        self.heads = nn.ModuleList([KBGraphAttentionalHead(
            entity_embedding_size_in,
            entity_embedding_size_out,
            relation_embedding_size,
            dropout_rate=dropout_rate) for _ in range(num_heads)])
        self.relation_fc = nn.Linear(
            relation_embedding_size, relation_embedding_size, bias=False)

        self.init_parameters()

    def init_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(
                self.relation_fc.weight, nonlinearity='relu')

    def reset_parameters(self):
        with torch.no_grad():
            for head in self.heads:
                head.reset_parameters()
            self.relation_fc.reset_parameters()

            self.init_parameters()

    def forward(
            self,
            entity_features,
            relation_features,
            triple_list,
            sparse_triple_adjacency_list_indices):
        with torch.no_grad():
            head_entities = triple_list[..., 0]
            tail_entities = triple_list[..., 1]
            relations = triple_list[..., 2]

        head_features = F.embedding(head_entities, entity_features)
        tail_features = F.embedding(tail_entities, entity_features)
        edge_features = F.embedding(relations, relation_features)
        del head_entities, tail_entities, relations

        triple_features = torch.cat(
            (head_features, tail_features, edge_features), -1)
        del head_features, tail_features, edge_features

        entity_features = [
            head(triple_features, sparse_triple_adjacency_list_indices)
            for head in self.heads]
        del triple_features

        if self.concatenate:
            entity_features = torch.cat(entity_features, -1)
        else:
            entity_features = torch.mean(
                torch.stack(entity_features, -1), -1)

        relation_features = self.relation_fc(relation_features)

        return entity_features, relation_features


class KBGraphAttentionalHead(nn.Module):
    def __init__(
            self,
            entity_embedding_size_in,
            entity_embedding_size_out,
            relation_embedding_size,
            dropout_rate=0.1):
        super(KBGraphAttentionalHead, self).__init__()
        self.entity_embedding_size_in = entity_embedding_size_in
        self.entity_embedding_size_out = entity_embedding_size_out
        self.relation_embedding_size = relation_embedding_size
        self.dropout_rate = dropout_rate

        self.triple_fc = nn.Linear(
            entity_embedding_size_in * 2 + relation_embedding_size,
            entity_embedding_size_out,
            bias=False)
        self.attention_fc = nn.Linear(entity_embedding_size_out, 1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

        self.init_parameters()

    def init_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(
                self.triple_fc.weight, nonlinearity='linear')
            nn.init.kaiming_uniform_(
                self.attention_fc.weight, nonlinearity='relu')

    def reset_parameters(self):
        with torch.no_grad():
            self.triple_fc.reset_parameters()
            self.attention_fc.reset_parameters()

            self.init_parameters()

    def forward(
            self,
            triple_features,
            sparse_triple_adjacency_list_indices):
        triple_features = self.dropout(triple_features)
        triple_features = self.triple_fc(triple_features)

        sparse_triple_features_adjacency_list = torch.sparse_coo_tensor(
            sparse_triple_adjacency_list_indices,
            triple_features,
            device=triple_features.device).coalesce()

        triple_attention = self.attention_fc(triple_features)
        triple_attention = torch.squeeze(triple_attention, dim=-1)
        triple_attention = F.mish(triple_attention)
        triple_attention = self.dropout(triple_attention)

        sparse_triple_attention_adjacency_list = torch.sparse_coo_tensor(
            sparse_triple_adjacency_list_indices,
            triple_attention,
            device=triple_attention.device)
        sparse_triple_attention_adjacency_list = torch.sparse.softmax(
            sparse_triple_attention_adjacency_list,
            dim=sparse_triple_attention_adjacency_list.dim() - 1).coalesce()

        triple_features = sparse_triple_features_adjacency_list.values()
        triple_attention = torch.unsqueeze(
            sparse_triple_attention_adjacency_list.values(), -1)

        triple_weighted_features = triple_features * triple_attention
        del triple_features, triple_attention

        sparse_triple_weighted_features_adjacency_list = \
            torch.sparse_coo_tensor(
                sparse_triple_features_adjacency_list.indices(),
                triple_weighted_features,
                device=triple_weighted_features.device)

        entity_features = torch.sparse.sum(
            sparse_triple_weighted_features_adjacency_list,
            dim=sparse_triple_weighted_features_adjacency_list.dim() - 2)
        entity_features = entity_features.to_dense()

        return entity_features
