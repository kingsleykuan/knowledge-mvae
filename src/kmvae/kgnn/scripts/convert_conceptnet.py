import argparse
import csv
import json

import networkx as nx
from kmvae.kgnn.knowledge_graph_data import (KnowledgeGraph,
                                             add_special_entities_relations,
                                             convert_triples_to_ids,
                                             entities_relations_to_ids,
                                             get_entities_relations,
                                             split_triples)
from tqdm import tqdm

SYMMETRIC_RELATIONS = [
    'RelatedTo',
    'Synonym',
    'Antonym',
    'DistinctFrom',
    'LocatedNear',
    'SimilarTo',
    'EtymologicallyRelatedTo',
]
EXCLUDE_RELATIONS = [
    'Antonym',
    'DistinctFrom',
    'EtymologicallyRelatedTo',
    'EtymologicallyDerivedFrom',
    'ExternalURL',
    'dbpedia',
    'InstanceOf',
    'Entails',
    'NotDesires',
    'NotUsedFor',
    'NotCapableOf',
    'NotHasProperty',
]


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert ConceptNet dataset.")

    parser.add_argument(
        '--conceptnet_csv_path',
        default='data/conceptnet-assertions-5.7.0.csv', type=str,
        help='ConceptNet CSV path.')

    parser.add_argument(
        '--output_path', default='data/conceptnet/conceptnet', type=str,
        help='Output path.')

    parser.add_argument(
        '--output_train_path', default='data/conceptnet/train', type=str,
        help='Output train path.')

    parser.add_argument(
        '--output_val_path', default='data/conceptnet/val', type=str,
        help='Output validation path.')

    parser.add_argument(
        '--output_test_path', default='data/conceptnet/test', type=str,
        help='Output test path.')

    parser.add_argument(
        '--split', default=False, type=bool,
        help='Whether to split dataset into train/val/test splits.')

    parser.add_argument(
        '--language', default='en', type=str,
        help='Filter ConceptNet by language.')

    parser.add_argument(
        '--filter_weight', default=None, type=float,
        help='Filter ConceptNet by minimum relation weight.')

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


def convert_conceptnet(
        conceptnet_csv_path,
        output_path,
        output_train_path,
        output_val_path,
        output_test_path,
        split=False,
        language='en',
        filter_weight=None,
        special_entities=None,
        special_relations=None,
        train_split=0.8,
        val_split=0.1,
        random_seed=0):
    graph = import_conceptnet_graph(
        conceptnet_csv_path,
        language=language,
        exclude_relations=EXCLUDE_RELATIONS,
        filter_weight=filter_weight)
    print("Nodes: {}, Edges: {}".format(graph.order(), graph.size()))

    triples = graph.edges(keys=True)
    del graph
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


def import_conceptnet_graph(
        conceptnet_csv_path,
        language=None,
        exclude_relations=None,
        filter_weight=None):
    graph = nx.MultiDiGraph()

    with open(conceptnet_csv_path, newline='') as file:
        reader = csv.DictReader(
            file,
            delimiter='\t',
            lineterminator='\n',
            fieldnames=['uri', 'relation', 'start', 'end', 'json'])

        for row in tqdm(reader):
            start = row['start'].split('/')
            end = row['end'].split('/')

            # Filter by language
            if start[2] != language or end[2] != language:
                continue

            # Filter by relation type
            relation = row['relation'].split('/')[2]
            if exclude_relations:
                if relation in exclude_relations:
                    continue

            # Filter by relation weight
            if filter_weight:
                json_data = json.loads(row['json'])
                weight = json_data['weight']
                if weight < filter_weight:
                    continue

            start = start[3]
            end = end[3]

            graph.add_edge(start, end, key=relation)
            if relation in SYMMETRIC_RELATIONS:
                graph.add_edge(end, start, key=relation)

    return graph


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    convert_conceptnet(**vars(args))
