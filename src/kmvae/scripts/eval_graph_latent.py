import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader

from kmvae.data_utils import random_worker_init_fn, recursive_to_device
from kmvae.kgnn.triple_eval import get_triple_ranks
from kmvae.kgnn.vae_models import ConvEDecoder, KBGraphAttentionNetworkEncoder
from kmvae.mvae.models import (LatentClassifier,
                               VariationalAutoencoder)
from kmvae.optimus.tokenizers import (get_optimus_bert_tokenizer,
                                      get_optimus_gpt2_tokenizer)
from kmvae.sentence_data import (SentenceKnowledgeGraphDataset,
                                 SentenceTripleTestDataset)

SENTENCES_TRAIN_PATH = 'data/goemotions/data/train.json'
SENTENCES_EVAL_PATH = 'data/goemotions/data/test.json'

KNOWLEDGE_GRAPH_PATH = 'data/emotional-context/emotional-context'


KBGAT_ENCODER_PATH = 'models/kbgat_encoder_200_new'
CONVE_DECODER_PATH = 'models/conve_decoder_200_new'


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
    data_loader, data_loader_link_pred = load_data(
        SENTENCES_EVAL_PATH, KNOWLEDGE_GRAPH_PATH)

    data_loader_train = load_data_train(
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

        eval_graph_decoder(
            encoder_outputs_list, data_loader_link_pred, vae_decoder)

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

    dataset = SentenceKnowledgeGraphDataset(
        sentences_path,
        knowledge_graph_path,
        encoder_tokenizer,
        decoder_tokenizer,
        NUM_CLASSES,
        negative_ratio=1,
        filter_triples=False,
        bernoulli_trick=False)
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=16,
        collate_fn=dataset.get_collate_fn(),
        pin_memory=True,
        worker_init_fn=random_worker_init_fn,
        prefetch_factor=2,
        persistent_workers=True)

    dataset_link_pred = SentenceTripleTestDataset(
        sentences_path,
        knowledge_graph_path)
    data_loader_link_pred = DataLoader(
        dataset_link_pred,
        batch_size=4,
        shuffle=True,
        num_workers=16,
        collate_fn=dataset_link_pred.get_collate_fn(),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True)

    return data_loader, data_loader_link_pred


def load_data_train(sentences_path, knowledge_graph_path):
    encoder_tokenizer = get_optimus_bert_tokenizer()
    decoder_tokenizer = get_optimus_gpt2_tokenizer()

    dataset = SentenceKnowledgeGraphDataset(
        sentences_path,
        knowledge_graph_path,
        encoder_tokenizer,
        decoder_tokenizer,
        NUM_CLASSES,
        negative_ratio=1,
        filter_triples=False,
        bernoulli_trick=False)
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=16,
        collate_fn=dataset.get_collate_fn(),
        pin_memory=True,
        worker_init_fn=random_worker_init_fn,
        prefetch_factor=2,
        persistent_workers=True)

    return data_loader


def load_model():
    kbgat_encoder = KBGraphAttentionNetworkEncoder.from_pretrained(
        KBGAT_ENCODER_PATH)
    conve_decoder = ConvEDecoder.from_pretrained(CONVE_DECODER_PATH)

    vae_encoder = VariationalAutoencoder(kbgat_encoder, None)
    vae_decoder = VariationalAutoencoder(None, conve_decoder)
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
        graph_encoder_inputs = data['graph_encoder_inputs']

        outputs = vae_encoder(graph_encoder_inputs, None, calc_loss=False)

        encoder_outputs_list.append(outputs['encoder_outputs'])
        label_vector_list.append(label_vector)

    return encoder_outputs_list, label_vector_list


def eval_graph_decoder(
        encoder_outputs_list,
        data_loader_link_pred,
        vae_decoder):
    vae_decoder = vae_decoder.eval()

    locs = []
    logscales = []

    for encoder_outputs in encoder_outputs_list:
        locs.append(encoder_outputs['loc'])
        logscales.append(encoder_outputs['logscale'])

    del encoder_outputs_list

    locs = torch.cat(locs)
    logscales = torch.cat(logscales)

    ranks_list = []
    pbar = tqdm(data_loader_link_pred, desc="Evaluation Graph Decoder")
    for step, data in enumerate(pbar, start=1):
        data = recursive_to_device(data, device, non_blocking=True)

        sentence_index = data['sentence_index']
        triples = data['triples']
        triples_mask = data['triples_mask']

        loc = F.embedding(sentence_index, locs)
        logscale = F.embedding(sentence_index, logscales)

        encoder_outputs = {
            'loc': loc,
            'logscale': logscale,
        }

        graph_decoder_inputs = {
            'triples': triples,
        }

        outputs = vae_decoder(
            encoder_outputs, graph_decoder_inputs, calc_loss=False)

        triple_scores = outputs['decoder_outputs']['triple_scores']
        triple_scores = triple_scores * triples_mask

        ranks = get_triple_ranks(
            triple_scores, descending=True, method='average')
        ranks_list.append(ranks)

        if step >= 11751:
            break

    ranks = torch.cat(ranks_list, dim=0).float()
    reciprocal_ranks = 1 / ranks
    mean_rank = torch.mean(ranks).item()
    mean_reciprocal_rank = torch.mean(reciprocal_ranks).item()
    hits_1 = torch.mean((ranks <= 1).float()).item()
    hits_3 = torch.mean((ranks <= 3).float()).item()
    hits_10 = torch.mean((ranks <= 10).float()).item()

    print("Mean Rank: {0:.5g}, Mean Reciprocal Rank: {1:.5g}".format(
        mean_rank, mean_reciprocal_rank))
    print("Hits@1: {0:.5g}, Hits@3: {1:.5g}, Hits@10: {2:.5g}".format(
        hits_1, hits_3, hits_10))


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
