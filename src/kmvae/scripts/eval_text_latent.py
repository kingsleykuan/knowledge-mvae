import torch
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader

from kmvae.data_utils import recursive_to_device
from kmvae.mvae.models import LatentClassifier, VariationalAutoencoder
from kmvae.optimus.tokenizers import (get_optimus_bert_tokenizer,
                                      get_optimus_gpt2_tokenizer)
from kmvae.optimus.vae_models import OptimusBertEncoder, OptimusGPT2Decoder
from kmvae.sentence_data import SentenceDataset

SENTENCES_TRAIN_PATH = 'data/goemotions/data/train.json'
SENTENCES_EVAL_PATH = 'data/goemotions/data/test.json'

KNOWLEDGE_GRAPH_PATH = 'data/emotional-context/emotional-context'


OPTIMUS_BERT_PATH = 'models/optimus/bert_latent'
OPTIMUS_GPT2_PATH = 'models/optimus/gpt2_latent'


BATCH_SIZE = 64

NUM_CLASSES = 28

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
LATENT_SIZE = 512
HIDDEN_SIZE = 1024
DROPOUT_RATE = 0.1
LABEL_SMOOTHING = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    data_loader = load_data(
        SENTENCES_EVAL_PATH, KNOWLEDGE_GRAPH_PATH)

    data_loader_train = load_data(
        SENTENCES_TRAIN_PATH, KNOWLEDGE_GRAPH_PATH)

    (
        vae_encoder,
        vae_decoder,
        vae_passthrough,
        latent_classifier,
    ) = load_model()

    vae_encoder = vae_encoder.to(device)
    vae_decoder = vae_decoder.to(device)
    vae_passthrough = vae_passthrough.to(device)
    latent_classifier = latent_classifier.to(device)

    with torch.no_grad():
        encoder_outputs_list, label_vector_list = eval_encoder(
            data_loader,
            vae_encoder)

        eval_text_decoder(
            encoder_outputs_list, data_loader, vae_decoder)

        encoder_outputs_list_train, label_vector_list_train = eval_encoder(
            data_loader_train,
            vae_encoder)

    train_latent_classifier(
        encoder_outputs_list_train,
        label_vector_list_train,
        vae_passthrough,
        latent_classifier)

    with torch.no_grad():
        eval_latent_classifier(
            encoder_outputs_list,
            label_vector_list,
            vae_passthrough,
            latent_classifier)


def load_data(sentences_path, knowledge_graph_path):
    encoder_tokenizer = get_optimus_bert_tokenizer()
    decoder_tokenizer = get_optimus_gpt2_tokenizer()

    dataset = SentenceDataset(
        sentences_path,
        encoder_tokenizer,
        decoder_tokenizer,
        NUM_CLASSES)
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=16,
        collate_fn=dataset.get_collate_fn(),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True)

    return data_loader


def load_model():
    bert_encoder = OptimusBertEncoder.from_pretrained(
        OPTIMUS_BERT_PATH)
    gpt2_decoder = OptimusGPT2Decoder.from_pretrained(
        OPTIMUS_GPT2_PATH)

    vae_encoder = VariationalAutoencoder(bert_encoder, None)
    vae_decoder = VariationalAutoencoder(None, gpt2_decoder)
    vae_passthrough = VariationalAutoencoder(None, None)

    latent_classifier = LatentClassifier(
        LATENT_SIZE,
        HIDDEN_SIZE,
        NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        label_smoothing=LABEL_SMOOTHING)

    return (
        vae_encoder,
        vae_decoder,
        vae_passthrough,
        latent_classifier,
    )


def eval_encoder(data_loader, vae_encoder):
    vae_encoder = vae_encoder.eval()

    encoder_outputs_list = []
    label_vector_list = []

    pbar = tqdm(data_loader, desc="Evaluation Encoder")
    for step, data in enumerate(pbar, start=1):
        data = recursive_to_device(data, device, non_blocking=True)

        label_vector = data['label_vector'].float()
        text_encoder_inputs = data['text_encoder_inputs']

        outputs = vae_encoder(text_encoder_inputs, None, calc_loss=False)

        encoder_outputs_list.append(outputs['encoder_outputs'])
        label_vector_list.append(label_vector)

    return encoder_outputs_list, label_vector_list


def eval_text_decoder(
        encoder_outputs_list,
        data_loader,
        vae_decoder):
    vae_decoder = vae_decoder.eval()

    cross_entropy_list = []
    for i in range(10):
        cross_entropy = []

        pbar = tqdm(data_loader, desc="Evaluation Text Decoder")
        for step, data in enumerate(pbar, start=1):
            data = recursive_to_device(data, device, non_blocking=True)

            text_decoder_inputs = data['text_decoder_inputs']

            encoder_outputs = encoder_outputs_list[step - 1]

            outputs = vae_decoder(
                encoder_outputs, text_decoder_inputs, calc_loss=True)

            cross_entropy.append(outputs['decoder_outputs']['loss'])
        cross_entropy = torch.cat(cross_entropy)
        cross_entropy_list.append(cross_entropy)

    cross_entropy = torch.mean(torch.stack(cross_entropy_list, dim=-1), dim=-1)
    perplexity = torch.exp(cross_entropy)
    print(torch.mean(perplexity))


def train_latent_classifier(
        encoder_outputs_list,
        label_vector_list,
        vae_passthrough,
        latent_classifier):
    vae_passthrough = vae_passthrough.eval()
    latent_classifier = latent_classifier.train()

    parameter_dicts = latent_classifier.parameter_dicts()
    optimizer = optim.AdamW(
        parameter_dicts, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in tqdm(range(100)):
        for encoder_outputs, label_vector in \
                zip(encoder_outputs_list, label_vector_list):
            outputs = vae_passthrough(encoder_outputs, None, calc_loss=False)

            latent_z = outputs['vae_outputs']['latent_z'].detach()
            outputs = latent_classifier(
                latent_z,
                labels=label_vector,
                calc_loss=True)

            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def eval_latent_classifier(
        encoder_outputs_list,
        label_vector_list,
        vae_passthrough,
        latent_classifier):
    vae_passthrough = vae_passthrough.eval()
    latent_classifier = latent_classifier.eval()

    preds_list = []
    labels = []

    for i in range(100):
        preds = []
        for encoder_outputs, label_vector in \
                zip(encoder_outputs_list, label_vector_list):
            outputs = vae_passthrough(encoder_outputs, None, calc_loss=False)

            latent_z = outputs['vae_outputs']['latent_z'].detach()
            outputs = latent_classifier(latent_z, calc_loss=False)

            preds.append(torch.sigmoid(outputs['outputs']))

            if i == 0:
                labels.append(label_vector)
        preds = torch.cat(preds)
        preds_list.append(preds)

    preds = torch.round(torch.mean(torch.stack(preds_list, dim=-1), dim=-1))

    preds = preds.detach().cpu().numpy()
    labels = torch.cat(labels).detach().cpu().numpy()

    print(classification_report(labels, preds))
    print(f1_score(labels, preds, average='macro'))


if __name__ == '__main__':
    main()
