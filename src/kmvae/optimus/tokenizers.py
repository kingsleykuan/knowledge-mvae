from transformers import (BertTokenizer,
                          GPT2Tokenizer)


def get_optimus_bert_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    return tokenizer


def get_optimus_gpt2_tokenizer():
    tokenizer = OptimusGPT2Tokenizer.from_pretrained(
        'gpt2', add_prefix_space=True)

    # Line 844 Optimus/code/examples/big_ae/run_lm_vae_training.py
    special_tokens_dict = {
        'pad_token': '<PAD>',
        'bos_token': '<BOS>',
        'eos_token': '<EOS>'
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    # Optimus uses one token for the latent z
    tokenizer.model_max_length = tokenizer.model_max_length - 1

    return tokenizer


class OptimusGPT2Tokenizer(GPT2Tokenizer):
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # Line 448 Optimus/code/examples/big_ae/utils.py
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
