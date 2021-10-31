import argparse
import json
import random

import networkx as nx

from kmvae.kgnn.knowledge_graph_data import (KnowledgeGraph,
                                             add_special_entities_relations,
                                             convert_ids_to_triples,
                                             convert_triples_to_ids,
                                             entities_relations_to_ids,
                                             get_entities_relations,
                                             split_triples)
from kmvae.nrc_vad_emotional_intensity import NRCVADEmotionalIntensity


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Create Emotional Context Knowledge Graph.")

    parser.add_argument(
        '--knowledge_graph_path',
        default='data/conceptnet/conceptnet', type=str,
        help='Knowledge Graph path to load from.')

    parser.add_argument(
        '--nrc_valence_path', default='data/nrc-vad/v-scores.txt', type=str,
        help='NRC-VAD Valence path to load from.')

    parser.add_argument(
        '--nrc_arousal_path', default='data/nrc-vad/a-scores.txt', type=str,
        help='NRC-VAD Arousal path to load from.')

    parser.add_argument(
        '--nrc_dominance_path', default='data/nrc-vad/d-scores.txt', type=str,
        help='NRC-VAD Dominance path to load from.')

    parser.add_argument(
        '--sentences_train_path', default='data/train.json', type=str,
        help='Sentence dataset train path to load from.')

    parser.add_argument(
        '--sentences_val_path', default='data/val.json', type=str,
        help='Sentence dataset val path to load from.')

    parser.add_argument(
        '--sentences_test_path', default='data/test.json', type=str,
        help='Sentence dataset test path to load from.')

    parser.add_argument(
        '--sentences_train_output_path', default='data/train.json', type=str,
        help='Sentence dataset train path to save to.')

    parser.add_argument(
        '--sentences_val_output_path', default='data/val.json', type=str,
        help='Sentence dataset val path to save to.')

    parser.add_argument(
        '--sentences_test_output_path', default='data/test.json', type=str,
        help='Sentence dataset test path to save to.')

    parser.add_argument(
        '--output_path',
        default='data/emotional-context/emotional-context', type=str,
        help='Output path.')

    parser.add_argument(
        '--output_train_path',
        default='data/emotional-context/train', type=str,
        help='Output train path.')

    parser.add_argument(
        '--output_val_path',
        default='data/emotional-context/val', type=str,
        help='Output validation path.')

    parser.add_argument(
        '--output_test_path',
        default='data/emotional-context/test', type=str,
        help='Output test path.')

    parser.add_argument(
        '--synonym_relation', default='Synonym', type=str,
        help='Synonym relation in Knowledge Graph used to expand NRC-VAD.')

    parser.add_argument(
        '--neighborhood_limit', default=5, type=int,
        help='Emotional Context neighborhood limit per sentence entity.')

    parser.add_argument(
        '--split', default=False, type=bool,
        help='Whether to split dataset into train/val/test splits.')

    parser.add_argument(
        '--special_entities', nargs='*', default=None, type=str,
        help='Special entities to add.')

    parser.add_argument(
        '--special_relations', nargs='*', default=None, type=str,
        help='Special relations to add.')

    parser.add_argument(
        '--train_split', default=0.8, type=float,
        help='Train split ratio.')

    parser.add_argument(
        '--val_split', default=0.1, type=float,
        help='Validation split ratio.')

    parser.add_argument(
        '--random_seed', default=0, type=int,
        help='Random seed when shuffling and spliting ConceptNet.')

    return parser


def create_emotional_context_knowledge_graph(
        knowledge_graph_path,
        nrc_valence_path,
        nrc_arousal_path,
        nrc_dominance_path,
        sentences_train_path,
        sentences_val_path,
        sentences_test_path,
        sentences_train_output_path,
        sentences_val_output_path,
        sentences_test_output_path,
        output_path,
        output_train_path,
        output_val_path,
        output_test_path,
        synonym_relation='Synonym',
        neighborhood_limit=10,
        split=False,
        special_entities=None,
        special_relations=None,
        train_split=0.8,
        val_split=0.1,
        random_seed=0):
    knowledge_graph = KnowledgeGraph.load(knowledge_graph_path)
    nrc_vad = NRCVADEmotionalIntensity(
        nrc_valence_path,
        nrc_arousal_path,
        nrc_dominance_path)
    with open(sentences_train_path) as file:
        sentences_train = json.load(file)
    with open(sentences_val_path) as file:
        sentences_val = json.load(file)
    with open(sentences_test_path) as file:
        sentences_test = json.load(file)

    word_synonyms = get_synonyms_from_knowledge_graph(
        knowledge_graph, nrc_vad.words, synonym_relation)
    nrc_vad.add_synonyms(word_synonyms)

    sentences_train = create_emotional_context_subgraphs(
        knowledge_graph,
        nrc_vad,
        sentences_train,
        neighborhood_limit,
        random_seed=random_seed)
    sentences_val = create_emotional_context_subgraphs(
        knowledge_graph,
        nrc_vad,
        sentences_val,
        neighborhood_limit,
        random_seed=random_seed)
    sentences_test = create_emotional_context_subgraphs(
        knowledge_graph,
        nrc_vad,
        sentences_test,
        neighborhood_limit,
        random_seed=random_seed)

    triples = set()
    for sentence in sentences_train:
        triples.update(sentence['triples'])
    for sentence in sentences_val:
        triples.update(sentence['triples'])
    for sentence in sentences_test:
        triples.update(sentence['triples'])
    print("Triples: {}".format(len(triples)))

    entities, relations = get_entities_relations(triples)
    entities, relations = add_special_entities_relations(
        entities,
        relations,
        special_entities=special_entities,
        special_relations=special_relations)
    print("Entities: {}, Relations: {}".format(len(entities), len(relations)))

    entity_to_id, relation_to_id = entities_relations_to_ids(
        entities, relations)

    triples = convert_triples_to_ids(triples, entity_to_id, relation_to_id)

    knowledge_graph = KnowledgeGraph(triples, entity_to_id, relation_to_id)
    knowledge_graph.save(output_path)
    del knowledge_graph

    if split:
        triples_train, triples_val, triples_test = split_triples(
            triples,
            train_split=train_split,
            val_split=val_split,
            random_seed=random_seed)
        del triples

        print("Train: {}, Val: {}, Test: {}".format(
            len(triples_train), len(triples_val), len(triples_test)))

        knowledge_graph_train = KnowledgeGraph(
            triples_train, entity_to_id, relation_to_id)
        knowledge_graph_val = KnowledgeGraph(
            triples_val, entity_to_id, relation_to_id)
        knowledge_graph_test = KnowledgeGraph(
            triples_test, entity_to_id, relation_to_id)

        knowledge_graph_train.save(output_train_path)
        knowledge_graph_val.save(output_val_path)
        knowledge_graph_test.save(output_test_path)

    # Reindex entities and save data
    for sentence in sentences_train:
        sentence['entity_ids'] = [
            entity_to_id[entity] for entity in sentence['entities']]
        sentence['neighbor_ids'] = [
            entity_to_id[entity] for entity in sentence['neighbors']]
        sentence['triple_ids'] = list(convert_triples_to_ids(
            sentence['triples'], entity_to_id, relation_to_id))
    with open(sentences_train_output_path, 'w') as file:
        json.dump(sentences_train, file, indent=4)

    for sentence in sentences_val:
        sentence['entity_ids'] = [
            entity_to_id[entity] for entity in sentence['entities']]
        sentence['neighbor_ids'] = [
            entity_to_id[entity] for entity in sentence['neighbors']]
        sentence['triple_ids'] = list(convert_triples_to_ids(
            sentence['triples'], entity_to_id, relation_to_id))
    with open(sentences_val_output_path, 'w') as file:
        json.dump(sentences_val, file, indent=4)

    for sentence in sentences_test:
        sentence['entity_ids'] = [
            entity_to_id[entity] for entity in sentence['entities']]
        sentence['neighbor_ids'] = [
            entity_to_id[entity] for entity in sentence['neighbors']]
        sentence['triple_ids'] = list(convert_triples_to_ids(
            sentence['triples'], entity_to_id, relation_to_id))
    with open(sentences_test_output_path, 'w') as file:
        json.dump(sentences_test, file, indent=4)


def get_synonyms_from_knowledge_graph(
        knowledge_graph,
        words,
        synonym_relation):
    triples = knowledge_graph.graph.edges(keys=True)
    synonym_relation_id = knowledge_graph.relation_to_id[synonym_relation]
    synonym_triples = [
        triple for triple in triples if triple[2] == synonym_relation_id]
    synonym_graph = knowledge_graph.graph.edge_subgraph(synonym_triples)

    word_synonyms = []
    for word in words:
        entity = word.replace(' ', '_')
        if entity in knowledge_graph.entity_to_id:
            entity_id = knowledge_graph.entity_to_id[entity]
            try:
                # Note that synonyms are symmetric so only successors
                # need to be considered
                synonyms = synonym_graph.neighbors(entity_id)
            except nx.NetworkXError:
                continue

            synonyms = [
                knowledge_graph.id_to_entity[entity].replace('_', ' ')
                for entity in synonyms]

            for synonym in synonyms:
                word_synonyms.append((word, synonym))
    word_synonyms.sort()

    return word_synonyms


def create_emotional_context_subgraphs(
        knowledge_graph,
        nrc_vad,
        sentences,
        neighborhood_limit,
        random_seed=0):
    random.seed(random_seed)
    id_to_entity = knowledge_graph.id_to_entity
    id_to_relation = knowledge_graph.id_to_relation
    reverse_graph = knowledge_graph.graph.reverse()

    for sentence in sentences:
        entity_ids = sentence['entity_ids']
        sentence_triples = set()
        neighbors = set()
        for entity_id in entity_ids:
            out_triples = knowledge_graph.graph.edges(entity_id, keys=True)
            in_triples = reverse_graph.edges(entity_id, keys=True)

            out_triples = [tuple(triple) for triple in out_triples]
            in_triples = [(h, t, r) for (t, h, r) in in_triples]

            triple_scores = []
            for triple in out_triples:
                neighbor = id_to_entity[triple[1]].replace('_', ' ')
                triple_scores.append(
                    (triple, nrc_vad.get_emotional_intensity(neighbor)))
            for triple in in_triples:
                neighbor = id_to_entity[triple[0]].replace('_', ' ')
                triple_scores.append(
                    (triple, nrc_vad.get_emotional_intensity(neighbor)))

            random.shuffle(triple_scores)
            triple_scores.sort(key=lambda x: x[1], reverse=True)
            triple_scores = triple_scores[:neighborhood_limit]
            triples = list(zip(*triple_scores))[0]
            sentence_triples.update(triples)

        for triple in sentence_triples:
            neighbors.add(triple[0])
            neighbors.add(triple[1])
        neighbors = neighbors - set(entity_ids)
        sentence['neighbors'] = [
            id_to_entity[neighbor] for neighbor in neighbors]

        sentence['triples'] = list(convert_ids_to_triples(
            sentence_triples, id_to_entity, id_to_relation))

    return sentences


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    create_emotional_context_knowledge_graph(**vars(args))
