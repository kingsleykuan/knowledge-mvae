import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_utils import filter_dict, list_of_dicts_to_dict_of_lists
from .kgnn.knowledge_graph_data import KnowledgeGraph
from .kgnn.knowledge_graph_utils import (
    KnowledgeTripleCorrupter,
    add_context,
    convert_triple_list_to_sparse_triple_adjacency_list,
    index_entities)

PAD = '[pad]'
IDENTITY_RELATION = '[identity]'
CONTEXT_ENTITY = '[context]'
CONTEXT_RELATION = '[context]'
NEIGHBOR_CONTEXT_RELATION = '[neighbor_context]'


class SentenceDataset(Dataset):
    def __init__(
            self,
            sentences_path,
            encoder_tokenizer,
            decoder_tokenizer,
            num_classes):
        with open(sentences_path) as file:
            sentences = json.load(file)
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.num_classes = num_classes

        for sentence in sentences:
            label_vector = np.zeros(self.num_classes)
            for label_id in sentence['label_ids']:
                label_vector[label_id] = 1
            sentence['label_vector'] = label_vector

        keys = ['label_vector', 'text_encoder_inputs', 'text_decoder_inputs']
        self.sentences = [
            filter_dict(sentence, keys) for sentence in sentences]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    def get_collate_fn(self):
        def collate_fn(batch):
            batch = list_of_dicts_to_dict_of_lists(batch)

            batch['label_vector'] = torch.tensor(batch['label_vector'])
            batch['text_encoder_inputs'] = self.encoder_tokenizer.pad(
                batch['text_encoder_inputs'],
                padding=True,
                return_tensors='pt')
            batch['text_decoder_inputs'] = self.decoder_tokenizer.pad(
                batch['text_decoder_inputs'],
                padding=True,
                return_tensors='pt')

            return batch
        return collate_fn


class SentenceKnowledgeGraphDataset(Dataset):
    def __init__(
            self,
            sentences_path,
            knowledge_graph_path,
            encoder_tokenizer,
            decoder_tokenizer,
            num_classes,
            negative_ratio=1,
            filter_triples=True,
            bernoulli_trick=True):
        self.sentences_path = sentences_path
        with open(sentences_path) as file:
            sentences = json.load(file)

        self.knowledge_graph_path = knowledge_graph_path
        self.knowledge_graph = KnowledgeGraph.load(knowledge_graph_path)

        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

        self.num_classes = num_classes
        self.negative_ratio = negative_ratio
        self.filter_triples = filter_triples
        self.bernoulli_trick = bernoulli_trick

        self.num_entities = self.knowledge_graph.num_entities
        self.num_relations = self.knowledge_graph.num_relations

        self.pad_id = self.knowledge_graph.entity_to_id[
            PAD]
        self.identity_relation_id = self.knowledge_graph.relation_to_id[
            IDENTITY_RELATION]
        self.context_entity_id = self.knowledge_graph.entity_to_id[
            CONTEXT_ENTITY]
        self.context_relation_id = self.knowledge_graph.relation_to_id[
            CONTEXT_RELATION]
        self.neighbor_context_relation_id = \
            self.knowledge_graph.relation_to_id[NEIGHBOR_CONTEXT_RELATION]

        self.knowledge_triple_corrupter = KnowledgeTripleCorrupter(
            self.num_entities,
            self.num_relations,
            negative_ratio=negative_ratio,
            filter_triples=filter_triples,
            bernoulli_trick=bernoulli_trick,
            triples=self.knowledge_graph.triples)
        if bernoulli_trick:
            self.knowledge_triple_corrupter.bernoulli_probabilities[
                self.context_relation_id] = 1.0
            self.knowledge_triple_corrupter.bernoulli_probabilities[
                self.neighbor_context_relation_id] = 1.0

        for sentence in sentences:
            label_vector = np.zeros(self.num_classes)
            for label_id in sentence['label_ids']:
                label_vector[label_id] = 1
            sentence['label_vector'] = label_vector

        filtered_sentences = [
            sentence for sentence in sentences
            if len(sentence['triple_ids']) > 0]
        print("Filtered: {}".format(len(sentences) - len(filtered_sentences)))
        sentences = filtered_sentences

        keys = [
            'label_vector',
            'text_encoder_inputs',
            'text_decoder_inputs',
            'entity_ids',
            'neighbor_ids',
            'triple_ids',
        ]
        self.sentences = [
            filter_dict(sentence, keys) for sentence in sentences]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        sentence_entity_ids = sentence['entity_ids']
        sentence_neighbor_ids = sentence['neighbor_ids']
        triple_ids = sentence['triple_ids']

        graph_encoder_inputs, graph_decoder_inputs = \
            self.preprocess_graph_data(
                sentence_entity_ids, sentence_neighbor_ids, triple_ids)

        sentence['graph_encoder_inputs'] = graph_encoder_inputs
        sentence['graph_decoder_inputs'] = graph_decoder_inputs

        return sentence

    def set_rng(self, rng):
        self.knowledge_triple_corrupter.set_rng(rng)

    def preprocess_graph_data(
            self,
            sentence_entity_ids,
            sentence_neighbor_ids,
            triple_ids):
        triples = list(triple_ids)
        triples, _ = add_context(
            triples,
            None,
            sentence_entity_ids,
            self.context_entity_id,
            self.context_relation_id,
            add_new_context_entity=False,
            use_context_index=False)
        triples, _ = add_context(
            triples,
            None,
            sentence_neighbor_ids,
            self.context_entity_id,
            self.neighbor_context_relation_id,
            add_new_context_entity=False,
            use_context_index=False)
        triples_set = set([tuple(triple) for triple in triples])
        triples = np.asarray(triples)
        # Corrupt triples from sentence context subgraph
        corrupted_triples = self.knowledge_triple_corrupter.corrupt_triples(
            triples, triple_filter_set=triples_set)
        triple_labels = np.concatenate(
            (
                np.ones(len(triples)),
                np.zeros(len(corrupted_triples)),
            ), 0)
        combined_triples = np.concatenate(
            (triples, corrupted_triples), 0)

        entity_ids = sentence_entity_ids + sentence_neighbor_ids
        triple_list = list(triple_ids)

        # Index entities to save computation
        # Computation will be done only on present entities
        entity_id_to_index = {
            entity_id: index
            for index, entity_id in enumerate(entity_ids)}
        triple_list = index_entities(triple_list, entity_id_to_index)
        sentence_entity_ids = [
            entity_id_to_index[entity_id]
            for entity_id in sentence_entity_ids]
        sentence_neighbor_ids = [
            entity_id_to_index[entity_id]
            for entity_id in sentence_neighbor_ids]

        # Add context entity to context subgraph
        # Context entity is appended to end of the entity id list
        # Add context relation from sentence entities to context entity
        triple_list, entity_ids = add_context(
            triple_list,
            entity_ids,
            sentence_entity_ids,
            self.context_entity_id,
            self.context_relation_id,
            add_new_context_entity=True,
            use_context_index=True)
        # Add context relation from neighbor entities to context entity
        triple_list, entity_ids = add_context(
            triple_list,
            entity_ids,
            sentence_neighbor_ids,
            self.context_entity_id,
            self.neighbor_context_relation_id,
            add_new_context_entity=False,
            use_context_index=True)
        context_entity_index = np.asarray(len(entity_ids) - 1)

        triple_list, sparse_triple_adjacency_list_indices = \
            convert_triple_list_to_sparse_triple_adjacency_list(
                triple_list,
                len(entity_ids),
                add_identity=True,
                identity_relation_id=self.identity_relation_id)

        entity_ids = np.asarray(entity_ids)
        relation_ids = np.arange(0, self.num_relations)

        graph_encoder_inputs = {
            'entity_ids': entity_ids,
            'relation_ids': relation_ids,
            'triple_list': triple_list,
            'sparse_triple_adjacency_list_indices':
                sparse_triple_adjacency_list_indices,
            'context_entity_index': context_entity_index,
        }

        graph_decoder_inputs = {
            'triples': combined_triples,
            'triple_labels': triple_labels,
        }

        return graph_encoder_inputs, graph_decoder_inputs

    def get_collate_fn(self):
        def collate_fn(batch):
            batch = list_of_dicts_to_dict_of_lists(batch)

            batch['label_vector'] = torch.tensor(batch['label_vector'])

            batch['text_encoder_inputs'] = self.encoder_tokenizer.pad(
                batch['text_encoder_inputs'],
                padding=True,
                return_tensors='pt')
            batch['text_decoder_inputs'] = self.decoder_tokenizer.pad(
                batch['text_decoder_inputs'],
                padding=True,
                return_tensors='pt')

            batch_size = len(batch['graph_encoder_inputs'])

            # Batch graph encoder inputs
            batch['graph_encoder_inputs'] = list_of_dicts_to_dict_of_lists(
                batch['graph_encoder_inputs'])
            entity_ids = batch['graph_encoder_inputs']['entity_ids']
            relation_ids = batch['graph_encoder_inputs']['relation_ids']
            triple_list = batch['graph_encoder_inputs']['triple_list']
            sparse_triple_adjacency_list_indices = (
                batch['graph_encoder_inputs']
                     ['sparse_triple_adjacency_list_indices'])
            context_entity_index = \
                batch['graph_encoder_inputs']['context_entity_index']

            entity_ids_offset = 0
            relation_ids_offset = 0
            for i in range(1, batch_size):
                entity_ids_offset += len(entity_ids[i - 1])
                relation_ids_offset += len(relation_ids[i - 1])

                triple_list[i][..., 0] += entity_ids_offset
                triple_list[i][..., 1] += entity_ids_offset
                triple_list[i][..., 2] += relation_ids_offset

                sparse_triple_adjacency_list_indices[i][0] += \
                    entity_ids_offset

                context_entity_index[i] += entity_ids_offset

            graph_encoder_inputs = {
                'entity_ids': torch.from_numpy(np.concatenate(
                    entity_ids, axis=0)),
                'relation_ids': torch.from_numpy(np.concatenate(
                    relation_ids, axis=0)),
                'triple_list': torch.from_numpy(np.concatenate(
                    triple_list, axis=0)),
                'sparse_triple_adjacency_list_indices':
                    torch.from_numpy(np.concatenate(
                        sparse_triple_adjacency_list_indices, axis=1)),
                'context_entity_index': torch.from_numpy(np.stack(
                    context_entity_index, axis=0)),
            }
            batch['graph_encoder_inputs'] = graph_encoder_inputs

            # Batch graph decoder inputs
            batch['graph_decoder_inputs'] = list_of_dicts_to_dict_of_lists(
                batch['graph_decoder_inputs'])
            triples = batch['graph_decoder_inputs']['triples']
            triple_labels = batch['graph_decoder_inputs']['triple_labels']
            loss_mask_batch = []

            max_len = max([len(t) for t in triples])
            for i in range(batch_size):
                loss_mask = np.ones(len(triples[i]))

                triples[i] = np.pad(
                    triples[i],
                    ((0, max_len - len(triples[i])), (0, 0)),
                    constant_values=self.pad_id)

                triple_labels[i] = np.pad(
                    triple_labels[i],
                    (0, max_len - len(triple_labels[i])),
                    constant_values=0)

                loss_mask = np.pad(
                    loss_mask,
                    (0, max_len - len(loss_mask)),
                    constant_values=0)
                loss_mask_batch.append(loss_mask)

            graph_decoder_inputs = {
                'triples': torch.from_numpy(np.stack(
                    triples, axis=0)),
                'triple_labels': torch.from_numpy(np.stack(
                    triple_labels, axis=0)),
                'loss_mask': torch.from_numpy(np.stack(
                    loss_mask_batch, axis=0)),
            }
            batch['graph_decoder_inputs'] = graph_decoder_inputs

            return batch
        return collate_fn


class SentenceTripleTestDataset(Dataset):
    def __init__(
            self,
            sentences_path,
            knowledge_graph_path):
        self.sentences_path = sentences_path
        with open(sentences_path) as file:
            sentences = json.load(file)

        self.knowledge_graph_path = knowledge_graph_path
        self.knowledge_graph = KnowledgeGraph.load(knowledge_graph_path)

        self.num_entities = self.knowledge_graph.num_entities
        self.num_relations = self.knowledge_graph.num_relations

        self.context_entity_id = self.knowledge_graph.entity_to_id[
            CONTEXT_ENTITY]
        self.context_relation_id = self.knowledge_graph.relation_to_id[
            CONTEXT_RELATION]
        self.neighbor_context_relation_id = \
            self.knowledge_graph.relation_to_id[NEIGHBOR_CONTEXT_RELATION]

        filtered_sentences = [
            sentence for sentence in sentences
            if len(sentence['triple_ids']) > 0]
        print("Filtered: {}".format(len(sentences) - len(filtered_sentences)))
        sentences = filtered_sentences

        for sentence in sentences:
            triples = sentence['triple_ids']
            triples, _ = add_context(
                triples,
                None,
                sentence['entity_ids'],
                self.context_entity_id,
                self.context_relation_id,
                add_new_context_entity=False,
                use_context_index=False)
            triples, _ = add_context(
                triples,
                None,
                sentence['neighbor_ids'],
                self.context_entity_id,
                self.neighbor_context_relation_id,
                add_new_context_entity=False,
                use_context_index=False)
            sentence['triple_ids'] = triples

        self.sentence_triples = []
        for sentence_index, sentence in enumerate(sentences):
            for triple in sentence['triple_ids']:
                self.sentence_triples.append({
                    'sentence_index': sentence_index,
                    'triple': triple,
                })

        self.triple_filter_sets = [
            set([tuple(triple) for triple in sentence['triple_ids']])
            for sentence in sentences]

    def __len__(self):
        return len(self.sentence_triples) * 2

    def __getitem__(self, idx):
        sentence_triple = self.sentence_triples[idx // 2]
        sentence_index = sentence_triple['sentence_index']
        triple = sentence_triple['triple']

        # Replace head or tail entity
        replace_index = (idx % 2)

        triples, triples_mask = self.generate_test_triples(
            triple, replace_index, sentence_index)

        test_triples = {
            'sentence_index': sentence_index,
            'triples': triples,
            'triples_mask': triples_mask,
        }

        return test_triples

    def generate_test_triples(self, triple, replace_index, sentence_index):
        test_triples = np.tile(triple, ((self.num_entities, 1)))
        test_triples[:, replace_index] = np.arange(self.num_entities)
        test_triples[triple[replace_index]] = test_triples[0]
        test_triples[0] = triple

        mask = np.asarray([
            np.nan if tuple(triple.tolist())
            in self.triple_filter_sets[sentence_index]
            else 1.0 for triple in test_triples])
        mask[0] = 1.0

        return test_triples, mask

    def get_collate_fn(self):
        def collate_fn(batch):
            batch = list_of_dicts_to_dict_of_lists(batch)

            batch['sentence_index'] = torch.from_numpy(np.asarray(
                batch['sentence_index']))
            batch['triples'] = torch.from_numpy(np.asarray(
                batch['triples']))
            batch['triples_mask'] = torch.from_numpy(np.asarray(
                batch['triples_mask']))

            return batch
        return collate_fn
