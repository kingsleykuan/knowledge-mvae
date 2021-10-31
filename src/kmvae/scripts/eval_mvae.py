import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader

from kmvae.data_utils import random_worker_init_fn, recursive_to_device
from kmvae.kgnn.triple_eval import get_triple_ranks
from kmvae.kgnn.vae_models import ConvEDecoder, KBGraphAttentionNetworkEncoder
from kmvae.mvae.models import (FeatureDecoder,
                               LatentClassifier,
                               MultimodalVariationalAutoencoder)
from kmvae.optimus.tokenizers import (get_optimus_bert_tokenizer,
                                      get_optimus_gpt2_tokenizer)
from kmvae.optimus.vae_models import OptimusBertEncoder, OptimusGPT2Decoder
from kmvae.sentence_data import (SentenceKnowledgeGraphDataset,
                                 SentenceTripleTestDataset)

SENTENCES_TRAIN_PATH = 'data/goemotions/data/train.json'
SENTENCES_EVAL_PATH = 'data/goemotions/data/test.json'

KNOWLEDGE_GRAPH_PATH = 'data/emotional-context/emotional-context'


LABEL_ENCODER_PATH = 'models/mvae_all/label_encoder'
LABEL_DECODER_PATH = 'models/mvae_all/label_decoder'

OPTIMUS_BERT_PATH = 'models/mvae_all/bert'
OPTIMUS_GPT2_PATH = 'models/mvae_all/gpt2'

KBGAT_ENCODER_PATH = 'models/mvae_all/kbgat_encoder_200'
CONVE_DECODER_PATH = 'models/mvae_all/conve_decoder_200'


BATCH_SIZE = 32

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
        mvae_encoder,
        mvae_label_decoder,
        mvae_text_decoder,
        mvae_graph_decoder,
        mvae_passthrough,
        latent_classifier,
    ) = load_model()

    mvae_encoder = mvae_encoder.to(device)
    mvae_label_decoder = mvae_label_decoder.to(device)
    mvae_text_decoder = mvae_text_decoder.to(device)
    mvae_graph_decoder = mvae_graph_decoder.to(device)
    mvae_passthrough = mvae_passthrough.to(device)
    latent_classifier = latent_classifier.to(device)

    with torch.no_grad():
        encoder_outputs_list, label_vector_list = eval_encoder(
            data_loader,
            mvae_encoder)

        eval_label_decoder(
            encoder_outputs_list, data_loader, mvae_label_decoder)

        eval_text_decoder(
            encoder_outputs_list, data_loader, mvae_text_decoder)

        eval_graph_decoder(
            encoder_outputs_list, data_loader_link_pred, mvae_graph_decoder)

        encoder_outputs_list_train, label_vector_list_train = eval_encoder(
            data_loader_train,
            mvae_encoder)

    train_latent_classifier(
        encoder_outputs_list_train,
        label_vector_list_train,
        mvae_passthrough,
        latent_classifier)

    with torch.no_grad():
        eval_latent_classifier(
            encoder_outputs_list,
            label_vector_list,
            mvae_passthrough,
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
        batch_size=1,
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
    label_decoder = FeatureDecoder.from_pretrained(LABEL_DECODER_PATH)

    bert_encoder = OptimusBertEncoder.from_pretrained(OPTIMUS_BERT_PATH)
    gpt2_decoder = OptimusGPT2Decoder.from_pretrained(OPTIMUS_GPT2_PATH)

    kbgat_encoder = KBGraphAttentionNetworkEncoder.from_pretrained(
        KBGAT_ENCODER_PATH)
    conve_decoder = ConvEDecoder.from_pretrained(CONVE_DECODER_PATH)

    mvae_encoder = MultimodalVariationalAutoencoder(
        (
            (bert_encoder, None),
            (kbgat_encoder, None),
        ),
        subsample=False)

    mvae_label_decoder = MultimodalVariationalAutoencoder(
        (
            (None, label_decoder),
            (None, None),
            (None, None),
        ),
        subsample=False)

    mvae_text_decoder = MultimodalVariationalAutoencoder(
        (
            (None, gpt2_decoder),
            (None, None),
        ),
        subsample=False)

    mvae_graph_decoder = MultimodalVariationalAutoencoder(
        (
            (None, None),
            (None, conve_decoder),
        ),
        subsample=False)

    mvae_passthrough = MultimodalVariationalAutoencoder(
        (
            (None, None),
            (None, None),
        ),
        subsample=False)

    latent_classifier = LatentClassifier(
        LATENT_SIZE,
        HIDDEN_SIZE,
        NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        label_smoothing=LABEL_SMOOTHING)

    return (
        mvae_encoder,
        mvae_label_decoder,
        mvae_text_decoder,
        mvae_graph_decoder,
        mvae_passthrough,
        latent_classifier,
    )


def eval_encoder(data_loader, mvae_encoder):
    mvae_encoder = mvae_encoder.eval()

    encoder_outputs_list = []
    label_vector_list = []

    pbar = tqdm(data_loader, desc="Evaluation Encoder")
    for step, data in enumerate(pbar, start=1):
        data = recursive_to_device(data, device, non_blocking=True)

        label_vector = data['label_vector'].float()
        text_encoder_inputs = data['text_encoder_inputs']
        graph_encoder_inputs = data['graph_encoder_inputs']

        outputs = mvae_encoder(
            (
                (text_encoder_inputs, None),
                (graph_encoder_inputs, None),
            ),
            calc_loss=False)

        encoder_outputs_list.append(outputs['encoder_outputs'])
        label_vector_list.append(label_vector)

    return encoder_outputs_list, label_vector_list


def eval_label_decoder(
        encoder_outputs_list,
        data_loader,
        mvae_label_decoder):
    mvae_label_decoder = mvae_label_decoder.eval()

    preds_list = []
    labels = []
    for i in range(10):
        preds = []

        pbar = tqdm(data_loader, desc="Evaluation Label Decoder")
        for step, data in enumerate(pbar, start=1):
            data = recursive_to_device(data, device, non_blocking=True)

            label_vector = data['label_vector'].float()

            encoder_outputs = encoder_outputs_list[step - 1]

            outputs = mvae_label_decoder(
                (
                    (None, None),
                    (encoder_outputs[0], None),
                    (encoder_outputs[1], None),
                ),
                calc_loss=False)

            preds.append(torch.sigmoid(
                outputs['decoder_outputs'][0]['features']))

            if i == 0:
                labels.append(label_vector)
        preds = torch.cat(preds)
        preds_list.append(preds)

    preds = torch.round(torch.mean(torch.stack(preds_list, dim=-1), dim=-1))

    preds = preds.detach().cpu().numpy()
    labels = torch.cat(labels).detach().cpu().numpy()

    print(classification_report(labels, preds))
    print(f1_score(labels, preds, average='macro'))


def eval_text_decoder(
        encoder_outputs_list,
        data_loader,
        mvae_text_decoder):
    mvae_text_decoder = mvae_text_decoder.eval()

    cross_entropy_list = []
    for i in range(10):
        cross_entropy = []

        pbar = tqdm(data_loader, desc="Evaluation Text Decoder")
        for step, data in enumerate(pbar, start=1):
            data = recursive_to_device(data, device, non_blocking=True)

            text_decoder_inputs = data['text_decoder_inputs']

            encoder_outputs = encoder_outputs_list[step - 1]

            outputs = mvae_text_decoder(
                (
                    (encoder_outputs[0], text_decoder_inputs),
                    (encoder_outputs[1], None),
                ),
                calc_loss=True)

            cross_entropy.append(outputs['decoder_outputs'][0]['loss'])
        cross_entropy = torch.cat(cross_entropy)
        cross_entropy_list.append(cross_entropy)

    cross_entropy = torch.mean(torch.stack(cross_entropy_list, dim=-1), dim=-1)
    perplexity = torch.exp(cross_entropy)
    print(torch.mean(perplexity))


def eval_graph_decoder(
        encoder_outputs_list,
        data_loader_link_pred,
        mvae_graph_decoder):
    mvae_graph_decoder = mvae_graph_decoder.eval()

    locs_0 = []
    logscales_0 = []

    locs_1 = []
    logscales_1 = []

    for encoder_outputs in encoder_outputs_list:
        locs_0.append(encoder_outputs[0]['loc'])
        logscales_0.append(encoder_outputs[0]['logscale'])

        locs_1.append(encoder_outputs[1]['loc'])
        logscales_1.append(encoder_outputs[1]['logscale'])

    del encoder_outputs_list

    locs_0 = torch.cat(locs_0)
    logscales_0 = torch.cat(logscales_0)

    locs_1 = torch.cat(locs_1)
    logscales_1 = torch.cat(logscales_1)

    ranks_list = []
    pbar = tqdm(data_loader_link_pred, desc="Evaluation Graph Decoder")
    for step, data in enumerate(pbar, start=1):
        data = recursive_to_device(data, device, non_blocking=True)

        sentence_index = data['sentence_index']
        triples = data['triples']
        triples_mask = data['triples_mask']

        loc_0 = F.embedding(sentence_index, locs_0)
        logscale_0 = F.embedding(sentence_index, logscales_0)

        loc_1 = F.embedding(sentence_index, locs_1)
        logscale_1 = F.embedding(sentence_index, logscales_1)

        encoder_outputs_0 = {
            'loc': loc_0,
            'logscale': logscale_0,
        }

        encoder_outputs_1 = {
            'loc': loc_1,
            'logscale': logscale_1,
        }

        graph_decoder_inputs = {
            'triples': triples,
        }

        outputs = mvae_graph_decoder(
            (
                (encoder_outputs_0, None),
                (encoder_outputs_1, graph_decoder_inputs),
            ),
            calc_loss=False)

        triple_scores = outputs['decoder_outputs'][1]['triple_scores']
        triple_scores = triple_scores * triples_mask

        ranks = get_triple_ranks(
            triple_scores, descending=True, method='average')
        ranks_list.append(ranks)

        if step >= 10000:
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
        mvae_passthrough,
        latent_classifier):
    mvae_passthrough = mvae_passthrough.eval()
    latent_classifier = latent_classifier.train()

    parameter_dicts = latent_classifier.parameter_dicts()
    optimizer = optim.AdamW(
        parameter_dicts, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in tqdm(range(100)):
        for encoder_outputs, label_vector in \
                zip(encoder_outputs_list, label_vector_list):
            outputs = mvae_passthrough(
                (
                    (encoder_outputs[0], None),
                    (encoder_outputs[1], None),
                ),
                calc_loss=False)

            latent_z = outputs['mvae_outputs']['latent_z'].detach()
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
        mvae_passthrough,
        latent_classifier):
    mvae_passthrough = mvae_passthrough.eval()
    latent_classifier = latent_classifier.eval()

    preds_list = []
    labels = []

    for i in range(100):
        preds = []
        for encoder_outputs, label_vector in \
                zip(encoder_outputs_list, label_vector_list):
            outputs = mvae_passthrough(
                (
                    (encoder_outputs[0], None),
                    (encoder_outputs[1], None),
                ),
                calc_loss=False)

            latent_z = outputs['mvae_outputs']['latent_z'].detach()
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
