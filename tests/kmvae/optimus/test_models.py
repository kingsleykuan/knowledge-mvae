import sys
sys.path.append('kmvae/optimus/Optimus/code')

import pytest
import torch
from kmvae.mvae.models import VariationalAutoencoder
from kmvae.optimus.models import OptimusBert, OptimusGPT2
from kmvae.optimus.Optimus.code.examples.big_ae.modules.vae import VAE
from kmvae.optimus.Optimus.code.pytorch_transformers import (
    BertForLatentConnector, GPT2ForLatentConnector)
from kmvae.optimus.tokenizers import (get_optimus_bert_tokenizer,
                                      get_optimus_gpt2_tokenizer)
from kmvae.optimus.vae_models import OptimusBertEncoder, OptimusGPT2Decoder

OPTIMUS_BERT_PRETRAINED_PATH = 'checkpoint-508523/checkpoint-encoder-508523'
OPTIMUS_GPT2_PRETRAINED_PATH = 'checkpoint-508523/checkpoint-decoder-508523'
OPTIMUS_VAE_BIN_PATH = 'checkpoint-508523/checkpoint-full-508523/training.bin'

SAMPLE_TEXTS = [
    'The quick brown fox jumps over the lazy dog.',

    '''We hold these truths to be self-evident, that all men are created equal,
    that they are endowed by their Creator with certain unalienable Rights,
    that among these are Life, Liberty and the pursuit of Happiness.--That to
    secure these rights, Governments are instituted among Men, deriving their
    just powers from the consent of the governed, --That whenever any Form of
    Government becomes destructive of these ends, it is the Right of the People
    to alter or to abolish it, and to institute new Government, laying its
    foundation on such principles and organizing its powers in such form, as
    to them shall seem most likely to effect their Safety and Happiness.''',

    '''We the peoples of the United Nations determined to save succeeding
    generations from the scourge of war, which twice in our lifetime has
    brought untold sorrow to mankind, and to reaffirm faith in fundamental
    human rights, in the dignity and worth of the human person, in the equal
    rights of men and women and of nations large and small, and to establish
    conditions under which justice and respect for the obligations arising
    from treaties and other sources of international law can be maintained,
    and to promote social progress and better standards of life in larger
    freedom, and for these ends to practice tolerance and live together in
    peace with one another as good neighbours, and to unite our strength to
    maintain international peace and security, and to ensure, by the
    acceptance of principles and the institution of methods, that armed
    force shall not be used, save in the common interest, and to employ
    international machinery for the promotion of the economic and social
    advancement of all peoples, have resolved to combine our efforts to
    accomplish these aims.''',
]


@pytest.fixture
def optimus_bert_tokenizer_inputs():
    tokenizer = get_optimus_bert_tokenizer()
    inputs = tokenizer(
        SAMPLE_TEXTS, return_tensors='pt', padding=True, truncation=True)
    return tokenizer, inputs


@pytest.fixture
def optimus_gpt2_tokenizer_inputs():
    tokenizer = get_optimus_gpt2_tokenizer()
    inputs = tokenizer(
        SAMPLE_TEXTS, return_tensors='pt', padding=True, truncation=True)
    return tokenizer, inputs


def test_optimus_bert(optimus_bert_tokenizer_inputs):
    tokenizer, inputs = optimus_bert_tokenizer_inputs

    with torch.no_grad():
        model_old = BertForLatentConnector.from_pretrained(
            OPTIMUS_BERT_PRETRAINED_PATH, latent_size=32)
        model_old = model_old.eval()
        input_ids = inputs['input_ids']
        # Line 100 Optimus/code/examples/big_ae/modules/vae.py
        attention_mask = (input_ids > 0).float()
        outputs_old = model_old(input_ids, attention_mask)

        model_new = OptimusBert.from_pretrained(
            OPTIMUS_BERT_PRETRAINED_PATH, latent_size=32)
        model_new = model_new.eval()
        outputs_new = model_new(**inputs, return_dict=False)

        for output_old, output_new in zip(outputs_old, outputs_new):
            assert torch.allclose(
                output_old, output_new, atol=1e-5, equal_nan=True)


def test_optimus_gpt2(optimus_gpt2_tokenizer_inputs):
    tokenizer, inputs = optimus_gpt2_tokenizer_inputs
    pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        labels = inputs['input_ids']
        latent_z = torch.normal(0, 1, (len(SAMPLE_TEXTS), 32))

        model_old = GPT2ForLatentConnector.from_pretrained(
            OPTIMUS_GPT2_PRETRAINED_PATH, latent_size=32)
        model_old = model_old.eval()
        outputs_old = model_old(
            input_ids=labels,
            past=latent_z,
            labels=labels,
            label_ignore=pad_token_id)
        predictions_old = torch.argmax(outputs_old[1], dim=-1)

        model_new = OptimusGPT2.from_pretrained(
            OPTIMUS_GPT2_PRETRAINED_PATH, latent_size=32)
        model_new = model_new.eval()
        outputs_new = model_new(
            input_ids=labels,
            past=latent_z,
            labels=labels,
            label_ignore=pad_token_id)
        predictions_new = torch.argmax(outputs_new[1], dim=-1)

        assert torch.equal(predictions_old, predictions_new)


def test_optimus_vae(
        optimus_bert_tokenizer_inputs,
        optimus_gpt2_tokenizer_inputs):
    encoder_tokenizer, encoder_inputs = optimus_bert_tokenizer_inputs
    decoder_tokenizer, decoder_inputs = optimus_gpt2_tokenizer_inputs
    pad_token_id = decoder_tokenizer.pad_token_id

    with torch.no_grad():
        encoder_model = BertForLatentConnector.from_pretrained(
            OPTIMUS_BERT_PRETRAINED_PATH, latent_size=32)
        encoder_model = encoder_model.eval()

        decoder_model = GPT2ForLatentConnector.from_pretrained(
            OPTIMUS_GPT2_PRETRAINED_PATH, latent_size=32)
        decoder_model = decoder_model.eval()

        class Args:
            def __init__(self):
                self.latent_size = 32
                self.device = None
                self.fb_mode = 0
                self.length_weighted_loss = False
                self.beta = 1.0
        args = Args()

        vae = VAE(
            encoder_model,
            decoder_model,
            encoder_tokenizer,
            decoder_tokenizer,
            args)
        checkpoint = torch.load(OPTIMUS_VAE_BIN_PATH)
        vae.load_state_dict(checkpoint['model_state_dict'])
        vae = vae.eval()

        bert_fea = vae.encoder(**encoder_inputs)[1]
        mu, logvar = vae.encoder.linear(bert_fea).chunk(2, -1)
        z = vae.reparameterize(mu, logvar, 1).squeeze(1)

        outputs_old = vae.decoder(
            input_ids=decoder_inputs['input_ids'],
            past=z,
            labels=decoder_inputs['input_ids'],
            label_ignore=pad_token_id)
        predictions_old = torch.argmax(outputs_old[1], dim=-1)

        encoder_model = OptimusBertEncoder(
            OPTIMUS_BERT_PRETRAINED_PATH, latent_size=32)
        encoder_model = encoder_model.eval()

        decoder_model = OptimusGPT2Decoder(
            OPTIMUS_GPT2_PRETRAINED_PATH,
            latent_size=32,
            latent_as_gpt_emb=True,
            latent_as_gpt_memory=True,
            pad_token_id=pad_token_id)
        decoder_model = decoder_model.eval()

        vae = VariationalAutoencoder(encoder_model, decoder_model)
        outputs_new = vae(encoder_inputs, decoder_inputs, calc_loss=False)
        predictions_new = torch.argmax(
            outputs_new['decoder_outputs']['logits'], dim=-1)

        # Sampling of latent z introduces slight differences
        match_ratio = (
            torch.sum(torch.eq(predictions_old, predictions_new))
            / torch.numel(predictions_old))
        assert match_ratio > 0.9


def test_optimus_vae_loss(
        optimus_bert_tokenizer_inputs,
        optimus_gpt2_tokenizer_inputs):
    encoder_tokenizer, encoder_inputs = optimus_bert_tokenizer_inputs
    decoder_tokenizer, decoder_inputs = optimus_gpt2_tokenizer_inputs
    pad_token_id = decoder_tokenizer.pad_token_id

    with torch.no_grad():
        encoder_model = BertForLatentConnector.from_pretrained(
            OPTIMUS_BERT_PRETRAINED_PATH, latent_size=32)
        encoder_model = encoder_model.eval()

        decoder_model = GPT2ForLatentConnector.from_pretrained(
            OPTIMUS_GPT2_PRETRAINED_PATH, latent_size=32)
        decoder_model = decoder_model.eval()

        class Args:
            def __init__(self):
                self.latent_size = 32
                self.device = None
                self.fb_mode = 1
                self.dim_target_kl = 0.5
                self.length_weighted_loss = False
                self.beta = 1.0
        args = Args()

        vae = VAE(
            encoder_model,
            decoder_model,
            encoder_tokenizer,
            decoder_tokenizer,
            args)
        checkpoint = torch.load(OPTIMUS_VAE_BIN_PATH)
        vae.load_state_dict(checkpoint['model_state_dict'])
        vae = vae.eval()

        reconstruction_loss_old, kl_loss_old, loss_old = vae(
            encoder_inputs['input_ids'], decoder_inputs['input_ids'])

        encoder_model = OptimusBertEncoder(
            OPTIMUS_BERT_PRETRAINED_PATH, latent_size=32)
        encoder_model = encoder_model.eval()

        decoder_model = OptimusGPT2Decoder(
            OPTIMUS_GPT2_PRETRAINED_PATH,
            latent_size=32,
            latent_as_gpt_emb=True,
            latent_as_gpt_memory=True,
            pad_token_id=pad_token_id)
        decoder_model = decoder_model.eval()

        vae = VariationalAutoencoder(
            encoder_model,
            decoder_model,
            kl_free_bits=0.5,
            beta=1.0,
            loss_reduction='none')
        outputs = vae(encoder_inputs, decoder_inputs, calc_loss=True)
        outputs = outputs['vae_outputs']
        loss_new = outputs['loss']
        reconstruction_loss_new = outputs['reconstruction_loss']
        kl_loss_new = outputs['kl_loss']

        # Sampling of latent z introduces slight differences
        assert torch.allclose(loss_old, loss_new, rtol=0.1)
        assert torch.allclose(
            reconstruction_loss_old, reconstruction_loss_new, rtol=0.1)
        assert torch.allclose(kl_loss_old, kl_loss_new)


def test_optimus_vae_parameter_dicts():
    encoder_model = OptimusBertEncoder(
            OPTIMUS_BERT_PRETRAINED_PATH, latent_size=32)

    decoder_model = OptimusGPT2Decoder(
        OPTIMUS_GPT2_PRETRAINED_PATH,
        latent_size=32,
        latent_as_gpt_emb=True,
        latent_as_gpt_memory=True)

    vae = VariationalAutoencoder(encoder_model, decoder_model)

    # Line 279 Optimus/code/examples/big_ae/run_lm_vae_training.py
    exclude_weight_decay = ['bias', 'LayerNorm.weight']
    params_old = set([
        id(param) for name, param in vae.named_parameters()
        if not any(exclude in name for exclude in exclude_weight_decay)])
    params_exclude_weight_decay_old = set([
        id(param) for name, param in vae.named_parameters()
        if any(exclude in name for exclude in exclude_weight_decay)])

    parameter_dicts = vae.parameter_dicts()
    params_new = set([
        id(param) for param_dict in parameter_dicts
        for param in param_dict['params']
        if param_dict.get('weight_decay', None) is None])
    params_exclude_weight_decay_new = set([
        id(param) for param_dict in parameter_dicts
        for param in param_dict['params']
        if param_dict.get('weight_decay', None) == 0.0])

    assert params_old == params_new
    assert params_exclude_weight_decay_old == params_exclude_weight_decay_new
