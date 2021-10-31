import json
from pathlib import Path

import networkx as nx
from sklearn.model_selection import train_test_split


class KnowledgeGraph:
    """
    Knowledge Graph containing entities and their relations.

    triples are in the form (head_entity, tail_entity, relation)
    """

    def __init__(self, triples, entity_to_id, relation_to_id):
        self.triples = set([
            (head_entity, tail_entity, relation)
            for (head_entity, tail_entity, relation) in triples])

        self.graph = nx.MultiDiGraph()
        self.graph.add_edges_from(self.triples)

        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        self.id_to_entity = {v: k for k, v in entity_to_id.items()}
        self.id_to_relation = {v: k for k, v in relation_to_id.items()}

        self.num_entities = len(entity_to_id)
        self.num_relations = len(relation_to_id)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        graph_path = path / 'graph.gexf'
        metadata_path = path / 'metadata.json'

        nx.write_gexf(self.graph, graph_path)

        metadata = {
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'entity_to_id': self.entity_to_id,
            'relation_to_id': self.relation_to_id,
        }

        with metadata_path.open('w') as file:
            json.dump(metadata, file, indent=4)

    @classmethod
    def load(cls, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        graph_path = path / 'graph.gexf'
        metadata_path = path / 'metadata.json'

        graph = nx.read_gexf(graph_path, node_type=int)

        with metadata_path.open('r') as file:
            metadata = json.load(file)

        triples = graph.edges(keys=True)
        entity_to_id = metadata['entity_to_id']
        relation_to_id = metadata['relation_to_id']
        knowledge_graph = cls(triples, entity_to_id, relation_to_id)

        return knowledge_graph


def get_entities_relations(triples):
    head_entities, tail_entities, relations = zip(*triples)

    entities = set()
    entities.update(head_entities)
    entities.update(tail_entities)
    entities = list(entities)

    relations = list(set(relations))

    entities.sort()
    relations.sort()

    return entities, relations


def add_special_entities_relations(
        entities,
        relations,
        special_entities=None,
        special_relations=None):
    if special_entities:
        entities = list(special_entities) + list(entities)

    if special_relations:
        relations = list(special_relations) + list(relations)

    return entities, relations


def entities_relations_to_ids(entities, relations):
    entity_to_id = {entity: id for id, entity in enumerate(entities)}
    relation_to_id = {relation: id for id, relation in enumerate(relations)}

    return entity_to_id, relation_to_id


def convert_triples_to_ids(triples, entity_to_id, relation_to_id):
    triples = set([
        (
            entity_to_id[head_entity],
            entity_to_id[tail_entity],
            relation_to_id[relation],
        )
        for (head_entity, tail_entity, relation) in triples])
    return triples


def convert_ids_to_triples(triples, id_to_entity, id_to_relation):
    triples = set([
        (
            id_to_entity[head_entity],
            id_to_entity[tail_entity],
            id_to_relation[relation],
        )
        for (head_entity, tail_entity, relation) in triples])
    return triples


def split_triples(
        triples,
        train_split=0.8,
        val_split=0.1,
        random_seed=0):
    triples = [
        (head_entity, tail_entity, relation)
        for (head_entity, tail_entity, relation) in triples]
    train, val_test = train_test_split(
        triples,
        train_size=train_split,
        random_state=random_seed,
        shuffle=True)
    val, test = train_test_split(
        val_test,
        train_size=val_split / (1.0 - train_split),
        random_state=random_seed,
        shuffle=True)

    return train, val, test
