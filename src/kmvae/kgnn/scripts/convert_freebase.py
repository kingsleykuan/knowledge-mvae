import argparse

from kmvae.kgnn.knowledge_graph_data import (KnowledgeGraph,
                                             add_special_entities_relations,
                                             convert_triples_to_ids,
                                             entities_relations_to_ids,
                                             get_entities_relations)


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert FB15K-237 dataset.")

    parser.add_argument(
        '--freebase_train_path', default='data/FB15K-237/train.txt', type=str,
        help='FB15K-237 train path.')

    parser.add_argument(
        '--freebase_val_path', default='data/FB15K-237/valid.txt', type=str,
        help='FB15K-237 validation path.')

    parser.add_argument(
        '--freebase_test_path', default='data/FB15K-237/test.txt', type=str,
        help='FB15K-237 test path.')

    parser.add_argument(
        '--output_train_path', default='data/freebase/train', type=str,
        help='Output train path.')

    parser.add_argument(
        '--output_val_path', default='data/freebase/val', type=str,
        help='Output validation path.')

    parser.add_argument(
        '--output_test_path', default='data/freebase/test', type=str,
        help='Output test path.')

    parser.add_argument(
        '--special_entities', nargs='*', default=None, type=str,
        help='Special entities to add.')

    parser.add_argument(
        '--special_relations', nargs='*', default=None, type=str,
        help='Special relations to add.')

    return parser


def convert_freebase(
        freebase_train_path,
        freebase_val_path,
        freebase_test_path,
        output_train_path,
        output_val_path,
        output_test_path,
        special_entities=None,
        special_relations=None):
    triples_train = load_freebase_triples(freebase_train_path)
    triples_val = load_freebase_triples(freebase_val_path)
    triples_test = load_freebase_triples(freebase_test_path)

    triples = set()
    triples.update(triples_train)
    triples.update(triples_val)
    triples.update(triples_test)
    print("Triples: {}".format(len(triples)))
    print("Train: {}, Val: {}, Test: {}".format(
        len(triples_train), len(triples_val), len(triples_test)))

    entities, relations = get_entities_relations(triples)
    entities, relations = add_special_entities_relations(
        entities,
        relations,
        special_entities=special_entities,
        special_relations=special_relations)
    print("Entities: {}, Relations: {}".format(len(entities), len(relations)))

    entity_to_id, relation_to_id = entities_relations_to_ids(
        entities, relations)

    triples_train = convert_triples_to_ids(
        triples_train, entity_to_id, relation_to_id)
    triples_val = convert_triples_to_ids(
        triples_val, entity_to_id, relation_to_id)
    triples_test = convert_triples_to_ids(
        triples_test, entity_to_id, relation_to_id)

    knowledge_graph_train = KnowledgeGraph(
        triples_train, entity_to_id, relation_to_id)
    knowledge_graph_val = KnowledgeGraph(
        triples_val, entity_to_id, relation_to_id)
    knowledge_graph_test = KnowledgeGraph(
        triples_test, entity_to_id, relation_to_id)

    knowledge_graph_train.save(output_train_path)
    knowledge_graph_val.save(output_val_path)
    knowledge_graph_test.save(output_test_path)


def load_freebase_triples(path):
    triples = set()
    with open(path) as file:
        for line in file:
            head_entity, relation, tail_entity = line.rstrip().split('\t')
            triples.add((head_entity, tail_entity, relation))
    return triples


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    convert_freebase(**vars(args))
