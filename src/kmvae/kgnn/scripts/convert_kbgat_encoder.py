import argparse

import torch

from ..knowledge_graph_data import KnowledgeGraph
from ..vae_models import KBGraphAttentionNetworkEncoder


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert KBGAT to KBGAT encoder model.")

    parser.add_argument(
        '--knowledge_graph_path',
        default='data/emotional-context/emotional-context', type=str,
        help='Path to knowledge graph that model is trained on.')

    parser.add_argument(
        '--pretrained_kbgat_path', default=None, type=str,
        help='Path to load pretrained KBGAT model from.')

    parser.add_argument(
        '--kbgat_encoder_path', default='models/kbgat_encoder', type=str,
        help='Path to save KBGAT encoder to.')

    parser.add_argument(
        '--embedding_size', default=100, type=int, help='Embedding size.')

    parser.add_argument(
        '--num_heads', default=2, type=int, help='Number of heads in KBGAT.')

    parser.add_argument(
        '--latent_size', default=100, type=int, help='VAE latent size.')

    parser.add_argument(
        '--dropout_rate', default=0.1, type=float, help='Model dropout rate.')

    return parser


def convert_kbgat_encoder(
        knowledge_graph_path,
        pretrained_kbgat_path,
        kbgat_encoder_path,
        embedding_size,
        num_heads,
        latent_size,
        dropout_rate):
    knowledge_graph = KnowledgeGraph.load(knowledge_graph_path)

    if pretrained_kbgat_path:
        kbgat_state_dict = torch.load(
            pretrained_kbgat_path, map_location=torch.device('cpu'))
    else:
        kbgat_state_dict = None

    kbgat = KBGraphAttentionNetworkEncoder(
        knowledge_graph.num_entities,
        knowledge_graph.num_relations,
        embedding_size,
        num_heads,
        latent_size,
        dropout_rate=dropout_rate,
        kbgat_state_dict=kbgat_state_dict)
    kbgat = kbgat.eval()

    kbgat.save_pretrained(kbgat_encoder_path)


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    convert_kbgat_encoder(**vars(args))
