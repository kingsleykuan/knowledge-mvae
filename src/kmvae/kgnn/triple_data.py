"""Knowledge Graph Triple Dataset."""

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from .knowledge_graph_data import KnowledgeGraph
from .knowledge_graph_utils import (
    KnowledgeTripleCorrupter,
    add_context,
    convert_triple_list_to_sparse_triple_adjacency_list,
    find_local_subgraph_triples,
    reverse_graph_triples)

IDENTITY_RELATION = '[identity]'
CONTEXT_ENTITY = '[context]'
CONTEXT_RELATION = '[context]'


def random_worker_init_fn(worker_id):
    worker_info = get_worker_info()
    worker_info.dataset.set_rng(np.random.default_rng(worker_info.seed))


class TripleTrainDataset(Dataset):
    """Dataset of knowledge graph triples/corrupted triples for training."""

    def __init__(
            self,
            knowledge_graph_path,
            negative_ratio=1,
            filter_triples=True,
            bernoulli_trick=True):
        self.knowledge_graph = KnowledgeGraph.load(knowledge_graph_path)

        self.triples = np.asarray(list(self.knowledge_graph.triples))
        self.num_entities = self.knowledge_graph.num_entities
        self.num_relations = self.knowledge_graph.num_relations
        self.negative_ratio = negative_ratio
        self.filter_triples = filter_triples
        self.bernoulli_trick = bernoulli_trick

        self.knowledge_triple_corrupter = KnowledgeTripleCorrupter(
            self.num_entities,
            self.num_relations,
            negative_ratio=negative_ratio,
            filter_triples=filter_triples,
            bernoulli_trick=bernoulli_trick,
            triples=self.knowledge_graph.triples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]

    def set_rng(self, rng):
        self.knowledge_triple_corrupter.set_rng(rng)

    def get_collate_fn(self):
        def collate_fn(triples):
            triples = np.asarray(triples)
            corrupted_triples = self.corrupt_triples(triples)

            return (
                torch.from_numpy(triples),
                torch.from_numpy(corrupted_triples)
            )
        return collate_fn

    def corrupt_triples(self, triples):
        return self.knowledge_triple_corrupter.corrupt_triples(triples)


class TripleTestDataset(Dataset):
    """Dataset of possible knowledge graph triples for testing."""

    def __init__(
            self,
            knowledge_graph_path,
            filtered=False,
            all_knowledge_graph_paths=None):
        self.knowledge_graph = KnowledgeGraph.load(knowledge_graph_path)

        self.triples_test = np.asarray(list(self.knowledge_graph.triples))
        self.num_entities = self.knowledge_graph.num_entities
        self.num_relations = self.knowledge_graph.num_relations
        self.filtered = filtered

        if filtered:
            if all_knowledge_graph_paths is None:
                raise ValueError(
                    """all_knowledge_graph_paths must be provided
                    if filtered is True.""")

            self.all_triples_set = set()
            for path in all_knowledge_graph_paths:
                knowledge_graph = KnowledgeGraph.load(path)
                self.all_triples_set.update(knowledge_graph.triples)

        self.entities = np.arange(self.num_entities)

    def __len__(self):
        return len(self.triples_test) * 2

    def __getitem__(self, idx):
        triple = self.triples_test[idx // 2]

        # Replace head or tail entity
        replace_index = (idx % 2)

        return self.generate_test_triples(triple, replace_index)

    def generate_test_triples(self, triple, replace_index):
        test_triples = np.tile(triple, ((self.num_entities, 1)))
        test_triples[:, replace_index] = self.entities
        test_triples[triple[replace_index]] = test_triples[0]
        test_triples[0] = triple

        if self.filtered:
            mask = np.asarray([
                np.nan if tuple(triple.tolist()) in self.all_triples_set
                else 1.0 for triple in test_triples])
            mask[0] = 1.0

            return test_triples, mask
        else:
            return test_triples


class GraphTripleTrainDataset(TripleTrainDataset):
    def __init__(
            self,
            knowledge_graph_path,
            radius=2,
            negative_ratio=1,
            filter_triples=True,
            bernoulli_trick=True,
            add_context=False):
        super().__init__(
            knowledge_graph_path,
            negative_ratio=negative_ratio,
            filter_triples=filter_triples,
            bernoulli_trick=bernoulli_trick)
        self.radius = radius
        self.add_context = add_context

        self.graph = self.knowledge_graph.graph

        # Reverse graph so that local neighborhood search finds predecessors.
        self.reverse_graph = self.graph.reverse()

        self.identity_relation_id = self.knowledge_graph.relation_to_id[
            IDENTITY_RELATION]
        if self.add_context:
            self.context_entity_id = self.knowledge_graph.entity_to_id[
                CONTEXT_ENTITY]
            self.context_relation_id = self.knowledge_graph.relation_to_id[
                CONTEXT_RELATION]

    def get_collate_fn(self):
        def collate_fn(triples):
            triples = np.asarray(triples)
            corrupted_triples = self.corrupt_triples(triples)

            triple_entities = set()
            triple_entities.update(triples[:, 0])
            triple_entities.update(triples[:, 1])
            triple_entities.update(corrupted_triples[:, 0])
            triple_entities.update(corrupted_triples[:, 1])

            subgraph_triples, subgraph_entities = find_local_subgraph_triples(
                triple_entities,
                self.reverse_graph,
                self.radius,
                full_subgraph=False)
            entity_ids = list(subgraph_entities)
            triple_list = reverse_graph_triples(subgraph_triples)

            if self.add_context:
                triple_list, entity_ids = add_context(
                    triple_list,
                    entity_ids,
                    triple_entities,
                    self.context_entity_id,
                    self.context_relation_id)

            entity_id_to_index = {
                entity: i for i, entity in enumerate(entity_ids)}

            triple_list = [
                (entity_id_to_index[h], entity_id_to_index[t], r)
                for h, t, r in triple_list]
            triples = [
                (entity_id_to_index[h], entity_id_to_index[t], r)
                for h, t, r in triples]
            corrupted_triples = [
                (entity_id_to_index[h], entity_id_to_index[t], r)
                for h, t, r in corrupted_triples]

            triple_list, sparse_triple_adjacency_list_indices = \
                convert_triple_list_to_sparse_triple_adjacency_list(
                    triple_list,
                    len(entity_ids),
                    add_identity=True,
                    identity_relation_id=self.identity_relation_id)

            triples = np.asarray(triples)
            corrupted_triples = np.asarray(corrupted_triples)
            entity_ids = np.asarray(entity_ids)
            relation_ids = np.arange(0, self.num_relations)

            return (
                torch.from_numpy(triples),
                torch.from_numpy(corrupted_triples),
                torch.from_numpy(entity_ids),
                torch.from_numpy(relation_ids),
                torch.from_numpy(triple_list),
                torch.from_numpy(sparse_triple_adjacency_list_indices))
        return collate_fn


class GraphTripleTestDataset(TripleTestDataset):
    def __init__(
            self,
            knowledge_graph_test_path,
            knowledge_graph_train_path,
            radius=2,
            add_context=False,
            filtered=False,
            all_knowledge_graph_paths=None):
        super().__init__(
            knowledge_graph_test_path,
            filtered=filtered,
            all_knowledge_graph_paths=all_knowledge_graph_paths)
        self.radius = radius
        self.add_context = add_context

        self.knowledge_graph_train = KnowledgeGraph.load(
            knowledge_graph_train_path)
        self.graph = self.knowledge_graph_train.graph

        # Reverse graph so that local neighborhood search finds predecessors.
        self.reverse_graph = self.graph.reverse()

        self.identity_relation_id = self.knowledge_graph.relation_to_id[
            IDENTITY_RELATION]
        if self.add_context:
            self.context_entity_id = self.knowledge_graph.entity_to_id[
                CONTEXT_ENTITY]
            self.context_relation_id = self.knowledge_graph.relation_to_id[
                CONTEXT_RELATION]

    def get_collate_fn(self):
        def collate_fn(batch):
            if self.filtered:
                triples, mask = zip(*batch)
                mask = np.asarray(mask)
            else:
                triples = batch

            triples = np.asarray(triples)

            triples_flattened = np.reshape(triples, (-1, 3))
            triple_entities = set()
            triple_entities.update(triples_flattened[:, 0])
            triple_entities.update(triples_flattened[:, 1])

            subgraph_triples, subgraph_entities = find_local_subgraph_triples(
                triple_entities,
                self.reverse_graph,
                self.radius,
                full_subgraph=False)
            entity_ids = list(subgraph_entities)
            triple_list = reverse_graph_triples(subgraph_triples)

            if self.add_context:
                triple_list, entity_ids = add_context(
                    triple_list,
                    entity_ids,
                    triple_entities,
                    self.context_entity_id,
                    self.context_relation_id)

            entity_id_to_index = {
                entity: i for i, entity in enumerate(entity_ids)}

            triple_list = [
                (entity_id_to_index[h], entity_id_to_index[t], r)
                for h, t, r in triple_list]
            triples = [
                [
                    (entity_id_to_index[h], entity_id_to_index[t], r)
                    for h, t, r in triples_test]
                for triples_test in triples]

            triple_list, sparse_triple_adjacency_list_indices = \
                convert_triple_list_to_sparse_triple_adjacency_list(
                    triple_list,
                    len(entity_ids),
                    add_identity=True,
                    identity_relation_id=self.identity_relation_id)

            triples = np.asarray(triples)
            entity_ids = np.asarray(entity_ids)
            relation_ids = np.arange(0, self.num_relations)

            if self.filtered:
                return (
                    torch.from_numpy(triples),
                    torch.from_numpy(mask),
                    torch.from_numpy(entity_ids),
                    torch.from_numpy(relation_ids),
                    torch.from_numpy(triple_list),
                    torch.from_numpy(sparse_triple_adjacency_list_indices))
            else:
                return (
                    torch.from_numpy(triples),
                    torch.from_numpy(entity_ids),
                    torch.from_numpy(relation_ids),
                    torch.from_numpy(triple_list),
                    torch.from_numpy(sparse_triple_adjacency_list_indices))
        return collate_fn
