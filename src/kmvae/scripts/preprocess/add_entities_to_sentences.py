import argparse
import json

from kmvae.kgnn.entity_matcher import EntityMatcher
from kmvae.kgnn.knowledge_graph_data import KnowledgeGraph


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Add entities from knowledge graph to sentences.")

    parser.add_argument(
        '--knowledge_graph_path',
        default='data/conceptnet/conceptnet', type=str,
        help='Knowledge Graph path to load from.')

    parser.add_argument(
        '--load_sentences_path', default='data/train.json', type=str,
        help='Sentence dataset path to load from.')

    parser.add_argument(
        '--save_sentences_path', default='data/train.json', type=str,
        help='Sentence dataset path to save to.')

    return parser


def add_entities_to_sentences(
        knowledge_graph_path,
        load_sentences_path,
        save_sentences_path):
    knowledge_graph = KnowledgeGraph.load(knowledge_graph_path)
    entities = list(knowledge_graph.entity_to_id.keys())

    entity_matcher = EntityMatcher(entities, remove_stopwords=True)

    with open(load_sentences_path) as file:
        sentences = json.load(file)

    for sentence in sentences:
        matched_entities = entity_matcher.match_entities(
            sentence['text'], remove_stopwords=False)

        matched_entity_ids = [
            knowledge_graph.entity_to_id[entity]
            for entity in matched_entities]

        sentence['entities'] = matched_entities
        sentence['entity_ids'] = matched_entity_ids

    with open(save_sentences_path, 'w') as file:
        json.dump(sentences, file, indent=4)


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    add_entities_to_sentences(**vars(args))
