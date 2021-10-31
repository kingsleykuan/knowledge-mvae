import argparse

import torch

from ..knowledge_graph_data import KnowledgeGraph
from ..vae_models import ConvEDecoder


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert ConvE to ConvE encoder model.")

    parser.add_argument(
        '--knowledge_graph_path',
        default='data/emotional-context/emotional-context', type=str,
        help='Path to knowledge graph that model is trained on.')

    parser.add_argument(
        '--pretrained_knowledge_graph_embeddings_path', default=None, type=str,
        help='Path to load pretrained Knowledge Graph Embeddings from.')

    parser.add_argument(
        '--pretrained_conve_path', default=None, type=str,
        help='Path to load pretrained ConvE model from.')

    parser.add_argument(
        '--conve_decoder_path', default='models/conve_decoder', type=str,
        help='Path to save ConvE decoder to.')

    parser.add_argument(
        '--latent_size', default=100, type=int, help='VAE latent size.')

    parser.add_argument(
        '--embedding_size', default=100, type=int, help='Embedding size.')

    parser.add_argument(
        '--embedding_height', default=5, type=int,
        help='Embedding height in ConvE.')

    parser.add_argument(
        '--num_filters', default=32, type=int,
        help='Number of filters in ConvE.')

    parser.add_argument(
        '--dropout_rate', default=0.1, type=float, help='Model dropout rate.')

    parser.add_argument(
        '--label_smoothing', default=0.1, type=float, help='Label smoothing.')

    return parser


def convert_conve_decoder(
        knowledge_graph_path,
        pretrained_knowledge_graph_embeddings_path,
        pretrained_conve_path,
        conve_decoder_path,
        latent_size,
        embedding_size,
        embedding_height,
        num_filters,
        dropout_rate,
        label_smoothing):
    knowledge_graph = KnowledgeGraph.load(knowledge_graph_path)

    if pretrained_knowledge_graph_embeddings_path:
        knowledge_graph_embeddings_state_dict = torch.load(
            pretrained_knowledge_graph_embeddings_path,
            map_location=torch.device('cpu'))
    else:
        knowledge_graph_embeddings_state_dict = None

    if pretrained_conve_path:
        conve_state_dict = torch.load(
            pretrained_conve_path, map_location=torch.device('cpu'))
    else:
        conve_state_dict = None

    conve = ConvEDecoder(
        latent_size,
        knowledge_graph.num_entities,
        knowledge_graph.num_relations,
        embedding_size,
        embedding_height,
        num_filters,
        dropout_rate=dropout_rate,
        label_smoothing=label_smoothing,
        knowledge_graph_embeddings_state_dict=(
            knowledge_graph_embeddings_state_dict),
        conve_state_dict=conve_state_dict)
    conve = conve.eval()

    conve.save_pretrained(conve_decoder_path)


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    convert_conve_decoder(**vars(args))
