import networkx as nx
import numpy as np


class KnowledgeTripleCorrupter:
    def __init__(
            self,
            num_entities,
            num_relations,
            negative_ratio=1,
            filter_triples=False,
            bernoulli_trick=False,
            triples=None,
            rng=None):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.negative_ratio = negative_ratio
        self.filter_triples = filter_triples
        self.bernoulli_trick = bernoulli_trick

        if bernoulli_trick and triples is None:
            raise ValueError("""triples must be provided if
                bernoulli_trick is True""")

        if filter_triples and triples is not None:
            self.triple_filter_set = set(triples)

        if bernoulli_trick:
            self.bernoulli_probabilities = \
                calculate_triple_bernoulli_probabilities(
                    triples, num_entities, num_relations)

        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

    def set_rng(self, rng):
        self.rng = rng

    def corrupt_triples(self, triples, triple_filter_set=None):
        corrupted_triples = np.tile(triples, (self.negative_ratio, 1))

        if self.bernoulli_trick:
            bernoulli_probabilities = \
                self.bernoulli_probabilities[corrupted_triples[:, 2]]
            corrupt_indices = \
                (1 - self.rng.binomial(1, bernoulli_probabilities))
        else:
            corrupt_indices = self.rng.integers(
                0, 2, len(corrupted_triples))

        corrupt_entities = self.rng.integers(
            0, self.num_entities, len(corrupted_triples))

        for i in range(len(corrupted_triples)):
            corrupted_triples[i, corrupt_indices[i]] = corrupt_entities[i]

            while self.filter_triples:
                if triple_filter_set is None:
                    triple_filter_set = self.triple_filter_set

                if tuple(corrupted_triples[i]) not in triple_filter_set:
                    break
                else:
                    corrupt_entities[i] = self.rng.integers(
                        0, self.num_entities)
                    corrupted_triples[i, corrupt_indices[i]] = \
                        corrupt_entities[i]

        return corrupted_triples


def calculate_triple_bernoulli_probabilities(
        triples,
        num_entities,
        num_relations):
    """
    Calculate probability for corrupting head or tail of triple per relation.

    Follows "Knowledge Graph Embedding by Translating on Hyperplanes"
    Wang et al. (2014).
    """
    tails_per_heads = np.zeros((num_relations, num_entities))
    heads_per_tails = np.zeros((num_relations, num_entities))

    for triple in triples:
        head_entity, tail_entity, relation = triple
        tails_per_heads[relation, head_entity] += 1
        heads_per_tails[relation, tail_entity] += 1

    tails_per_heads[tails_per_heads == 0] = np.nan
    heads_per_tails[heads_per_tails == 0] = np.nan

    tails_per_heads = np.nanmean(tails_per_heads, axis=-1)
    heads_per_tails = np.nanmean(heads_per_tails, axis=-1)
    bernoulli_probabilities = tails_per_heads \
        / (tails_per_heads + heads_per_tails)
    return bernoulli_probabilities


def find_local_subgraph_triples(entities, graph, radius, full_subgraph=False):
    subgraph_entities = set(entities)

    if not full_subgraph:
        subgraph_triples = set()

    current = subgraph_entities
    for _ in range(radius):
        neighbors = set()
        for entity in current:
            try:
                neighbors.update(graph.neighbors(entity))
            except nx.NetworkXError:
                pass

        if not full_subgraph:
            neighborhood = graph.edges(current, keys=True)
            subgraph_triples.update(neighborhood)

        current = neighbors - subgraph_entities
        if len(current) <= 0:
            break

        subgraph_entities.update(current)

    if full_subgraph:
        subgraph_triples = graph.subgraph(subgraph_entities).edges(keys=True)
        subgraph_triples = set(subgraph_triples)

    return subgraph_triples, subgraph_entities


def reverse_graph_triples(triples):
    return [(h, t, r) for (t, h, r) in triples]


def index_entities(triples, entity_id_to_index):
    return [
        (entity_id_to_index[h], entity_id_to_index[t], r)
        for h, t, r in triples]


def add_context(
        triple_list,
        entity_list,
        add_context_entity_list,
        context_entity_id,
        context_relation_id,
        add_new_context_entity=True,
        use_context_index=True):
    if add_new_context_entity:
        entity_list.append(context_entity_id)

    if use_context_index:
        context_entity_id = len(entity_list) - 1
    for entity in add_context_entity_list:
        triple_list.append((entity, context_entity_id, context_relation_id))

    return triple_list, entity_list


def convert_triple_list_to_sparse_triple_adjacency_list(
        triple_list,
        num_entities,
        add_identity=True,
        identity_relation_id=0):
    if add_identity:
        for entity in range(num_entities):
            triple_list.append((entity, entity, identity_relation_id))

    triple_list = np.asarray(triple_list)
    num_triples_per_entity = [0 for _ in range(num_entities)]

    sparse_triple_adjacency_list_indices = np.full((2, len(triple_list)), -1)
    for i, triple in enumerate(triple_list):
        sparse_triple_adjacency_list_indices[0, i] = triple[1]
        sparse_triple_adjacency_list_indices[1, i] = \
            num_triples_per_entity[triple[1]]
        num_triples_per_entity[triple[1]] += 1

    return triple_list, sparse_triple_adjacency_list_indices
