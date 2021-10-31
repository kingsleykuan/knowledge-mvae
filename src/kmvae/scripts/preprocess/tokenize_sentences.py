import argparse
import json

from kmvae.optimus.tokenizers import (get_optimus_bert_tokenizer,
                                      get_optimus_gpt2_tokenizer)


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Tokenize sentences with encoder and decoder tokenizers.")

    parser.add_argument(
        '--load_sentences_path', default='data/train.json', type=str,
        help='Sentence dataset path to load from.')

    parser.add_argument(
        '--save_sentences_path', default='data/train.json', type=str,
        help='Sentence dataset path to save to.')

    parser.add_argument(
        '--max_token_length', default=128, type=int,
        help='Maximum token length after tokenizing.')

    return parser


def tokenize_sentences(
        load_sentences_path,
        save_sentences_path,
        max_token_length):
    with open(load_sentences_path) as file:
        sentences = json.load(file)

    encoder_tokenizer = get_optimus_bert_tokenizer()
    decoder_tokenizer = get_optimus_gpt2_tokenizer()

    encoder_tokenizer.model_max_length = max_token_length
    decoder_tokenizer.model_max_length = max_token_length

    for sentence in sentences:
        sentence['text_encoder_inputs'] = encoder_tokenizer(
            sentence['text'],
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=False).data
        sentence['text_decoder_inputs'] = decoder_tokenizer(
            sentence['text'],
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=False).data

    with open(save_sentences_path, 'w') as file:
        json.dump(sentences, file, indent=4)


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    tokenize_sentences(**vars(args))
