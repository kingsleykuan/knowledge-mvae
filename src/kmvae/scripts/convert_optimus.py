import argparse

from kmvae.optimus.tokenizers import get_optimus_gpt2_tokenizer
from kmvae.optimus.vae_models import OptimusBertEncoder, OptimusGPT2Decoder


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert Optimus model with new latent size.")

    parser.add_argument(
        '--optimus_bert_load_path',
        default='models/optimus/checkpoint-508523/checkpoint-encoder-508523',
        type=str,
        help='Path to load Optimus BERT from.')

    parser.add_argument(
        '--optimus_gpt2_load_path',
        default='models/optimus/checkpoint-508523/checkpoint-decoder-508523',
        type=str,
        help='Path to load Optimus GPT-2 from.')

    parser.add_argument(
        '--optimus_bert_save_path', default='models/optimus/bert', type=str,
        help='Path to save Optimus BERT to.')

    parser.add_argument(
        '--optimus_gpt2_save_path', default='models/optimus/gpt2', type=str,
        help='Path to save Optimus GPT-2 to.')

    parser.add_argument(
        '--original_latent_size', default=768, type=int,
        help='Original latent size.')

    parser.add_argument(
        '--new_latent_size', default=100, type=int,
        help='New latent size.')

    return parser


def convert_optimus(
        optimus_bert_load_path,
        optimus_gpt2_load_path,
        optimus_bert_save_path,
        optimus_gpt2_save_path,
        original_latent_size,
        new_latent_size):
    convert_optimus_bert(
        optimus_bert_load_path,
        optimus_bert_save_path,
        original_latent_size,
        new_latent_size)
    convert_optimus_gpt2(
        optimus_gpt2_load_path,
        optimus_gpt2_save_path,
        original_latent_size,
        new_latent_size)


def convert_optimus_bert(
        optimus_bert_load_path,
        optimus_bert_save_path,
        original_latent_size,
        new_latent_size):
    bert_encoder = OptimusBertEncoder(
        optimus_bert_load_path, latent_size=original_latent_size)

    bert_encoder.reset_latent(new_latent_size)
    bert_encoder.save_pretrained(optimus_bert_save_path)


def convert_optimus_gpt2(
        optimus_gpt2_load_path,
        optimus_gpt2_save_path,
        original_latent_size,
        new_latent_size):
    gpt2_tokenizer = get_optimus_gpt2_tokenizer()
    gpt2_decoder = OptimusGPT2Decoder(
        optimus_gpt2_load_path,
        latent_size=original_latent_size,
        latent_as_gpt_emb=True,
        latent_as_gpt_memory=True,
        pad_token_id=gpt2_tokenizer.pad_token_id)

    gpt2_decoder.reset_latent(new_latent_size)
    gpt2_decoder.save_pretrained(optimus_gpt2_save_path)


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    convert_optimus(**vars(args))
