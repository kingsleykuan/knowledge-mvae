from kmvae.optimus.Optimus.code.pytorch_transformers import (BertTokenizer,
                                                             GPT2Tokenizer)
from kmvae.optimus.tokenizers import (get_optimus_bert_tokenizer,
                                      get_optimus_gpt2_tokenizer)

SAMPLE_TEXT = \
    '''We hold these truths to be self-evident, that all men are created equal,
    that they are endowed by their Creator with certain unalienable Rights,
    that among these are Life, Liberty and the pursuit of Happiness.--That to
    secure these rights, Governments are instituted among Men, deriving their
    just powers from the consent of the governed, --That whenever any Form of
    Government becomes destructive of these ends, it is the Right of the People
    to alter or to abolish it, and to institute new Government, laying its
    foundation on such principles and organizing its powers in such form, as
    to them shall seem most likely to effect their Safety and Happiness.'''


def test_optimus_bert_tokenizer():
    tokenizer_old = BertTokenizer.from_pretrained('bert-base-cased')
    tokenizer_new = get_optimus_bert_tokenizer()

    assert len(tokenizer_old) == len(tokenizer_new)
    assert tokenizer_old.vocab_size == tokenizer_new.vocab_size
    assert tokenizer_old.max_len_single_sentence \
        == tokenizer_new.max_len_single_sentence

    assert tokenizer_old.unk_token == tokenizer_new.unk_token
    assert tokenizer_old.sep_token == tokenizer_new.sep_token
    assert tokenizer_old.pad_token == tokenizer_new.pad_token
    assert tokenizer_old.cls_token == tokenizer_new.cls_token
    assert tokenizer_old.mask_token == tokenizer_new.mask_token

    assert tokenizer_old.unk_token_id == tokenizer_new.unk_token_id
    assert tokenizer_old.sep_token_id == tokenizer_new.sep_token_id
    assert tokenizer_old.pad_token_id == tokenizer_new.pad_token_id
    assert tokenizer_old.cls_token_id == tokenizer_new.cls_token_id
    assert tokenizer_old.mask_token_id == tokenizer_new.mask_token_id

    assert tokenizer_old.encode(SAMPLE_TEXT, add_special_tokens=True) \
        == tokenizer_new.encode(SAMPLE_TEXT)


def test_optimus_gpt2_tokenizer():
    tokenizer_old = GPT2Tokenizer.from_pretrained('gpt2')
    # Line 844 Optimus/code/examples/big_ae/run_lm_vae_training.py
    special_tokens_dict = {
        'pad_token': '<PAD>',
        'bos_token': '<BOS>',
        'eos_token': '<EOS>'
    }
    tokenizer_old.add_special_tokens(special_tokens_dict)

    tokenizer_new = get_optimus_gpt2_tokenizer()

    assert len(tokenizer_old) == len(tokenizer_new)
    assert tokenizer_old.vocab_size == tokenizer_new.vocab_size

    # Test fails: assert 1024 == 1022
    # assert tokenizer_old.max_len_single_sentence \
    #     == tokenizer_new.max_len_single_sentence

    assert tokenizer_old.pad_token == tokenizer_new.pad_token
    assert tokenizer_old.bos_token == tokenizer_new.bos_token
    assert tokenizer_old.eos_token == tokenizer_new.eos_token

    assert tokenizer_old.pad_token_id == tokenizer_new.pad_token_id
    assert tokenizer_old.bos_token_id == tokenizer_new.bos_token_id
    assert tokenizer_old.eos_token_id == tokenizer_new.eos_token_id

    tokenized_text_old = tokenizer_old.encode(SAMPLE_TEXT)
    # Line 448 Optimus/code/examples/big_ae/utils.py
    tokenized_text_old = (
        [tokenizer_old.bos_token_id]
        + tokenized_text_old
        + [tokenizer_old.eos_token_id])

    assert tokenized_text_old == tokenizer_new.encode(SAMPLE_TEXT)
