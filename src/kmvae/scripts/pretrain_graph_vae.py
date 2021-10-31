import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from kmvae.data_utils import random_worker_init_fn, recursive_to_device
from kmvae.kgnn.triple_eval import get_triple_ranks
from kmvae.kgnn.vae_models import ConvEDecoder, KBGraphAttentionNetworkEncoder
from kmvae.mvae.models import LatentClassifier, VariationalAutoencoder
from kmvae.optimus.tokenizers import (get_optimus_bert_tokenizer,
                                      get_optimus_gpt2_tokenizer)
from kmvae.sentence_data import (SentenceKnowledgeGraphDataset,
                                 SentenceTripleTestDataset)
from kmvae.trainer import Trainer

SENTENCES_TRAIN_PATH = 'data/goemotions/data/train.json'
SENTENCES_VAL_PATH = 'data/goemotions/data/val.json'

KNOWLEDGE_GRAPH_PATH = 'data/emotional-context/emotional-context'

KBGAT_ENCODER_PATH = 'models_200/kbgat_encoder_new'
CONVE_DECODER_PATH = 'models_200/conve_decoder_new'

KBGAT_ENCODER_SAVE_PATH = 'models/kbgat_encoder_200_new'
CONVE_DECODER_SAVE_PATH = 'models/conve_decoder_200_new'

NUM_EPOCHS = 1000
STEPS_PER_LOG = 100
EPOCHS_PER_VAL = 10
MAX_EVAL_STEPS = 1000

BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

EVAL_BATCH_SIZE = 2

NEGATIVE_RATIO = 10

LATENT_SIZE = 512
HIDDEN_SIZE = 1024
NUM_CLASSES = 28
DROPOUT_RATE = 0.1
LABEL_SMOOTHING = 0.1


def main():
    data_loader_train = load_data_train(
        SENTENCES_TRAIN_PATH, KNOWLEDGE_GRAPH_PATH)
    data_loader_encoder_val, data_loader_decoder_val = load_data_val(
        SENTENCES_VAL_PATH, KNOWLEDGE_GRAPH_PATH)

    vae_train, vae_encoder_val, vae_decoder_val, latent_classifier = \
        load_model()

    trainer = GraphVAETrainer(
        [data_loader_train],
        [data_loader_encoder_val, data_loader_decoder_val],
        [vae_train, latent_classifier],
        [vae_encoder_val, vae_decoder_val, latent_classifier],
        NUM_EPOCHS,
        STEPS_PER_LOG,
        EPOCHS_PER_VAL,
        MAX_EVAL_STEPS,
        GRADIENT_ACCUMULATION_STEPS,
        LEARNING_RATE,
        WEIGHT_DECAY)

    trainer.train()


def load_data_train(sentences_path, knowledge_graph_path):
    encoder_tokenizer = get_optimus_bert_tokenizer()
    decoder_tokenizer = get_optimus_gpt2_tokenizer()

    dataset = SentenceKnowledgeGraphDataset(
        sentences_path,
        knowledge_graph_path,
        encoder_tokenizer,
        decoder_tokenizer,
        NUM_CLASSES,
        negative_ratio=NEGATIVE_RATIO,
        filter_triples=True,
        bernoulli_trick=True)
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        collate_fn=dataset.get_collate_fn(),
        pin_memory=True,
        worker_init_fn=random_worker_init_fn,
        prefetch_factor=2,
        persistent_workers=True)

    return data_loader


def load_data_val(sentences_path, knowledge_graph_path):
    encoder_tokenizer = get_optimus_bert_tokenizer()
    decoder_tokenizer = get_optimus_gpt2_tokenizer()

    dataset_encoder = SentenceKnowledgeGraphDataset(
        sentences_path,
        knowledge_graph_path,
        encoder_tokenizer,
        decoder_tokenizer,
        NUM_CLASSES,
        negative_ratio=1,
        filter_triples=False,
        bernoulli_trick=False)
    data_loader_encoder = DataLoader(
        dataset_encoder,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=16,
        collate_fn=dataset_encoder.get_collate_fn(),
        pin_memory=True,
        worker_init_fn=random_worker_init_fn,
        prefetch_factor=2,
        persistent_workers=True)

    dataset_decoder = SentenceTripleTestDataset(
        sentences_path,
        knowledge_graph_path)
    data_loader_decoder = DataLoader(
        dataset_decoder,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        collate_fn=dataset_decoder.get_collate_fn(),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True)

    return data_loader_encoder, data_loader_decoder


def load_model():
    kbgat_encoder = KBGraphAttentionNetworkEncoder.from_pretrained(
        KBGAT_ENCODER_PATH)

    conve_decoder = ConvEDecoder.from_pretrained(CONVE_DECODER_PATH)

    vae_train = VariationalAutoencoder(
        kbgat_encoder,
        conve_decoder,
        kl_free_bits=2.0,
        beta_schedule='cyclical',
        beta_start=0.0,
        beta_stop=1.0,
        beta_cyclical_total_steps=2702000,
        beta_cycles=100,
        beta_cycle_ratio_zero=0.5,
        beta_cycle_ratio_increase=0.25,
        loss_reduction='mean')

    vae_encoder_val = VariationalAutoencoder(
        kbgat_encoder,
        None)

    vae_decoder_val = VariationalAutoencoder(
        None,
        conve_decoder)

    latent_classifier = LatentClassifier(
        LATENT_SIZE,
        HIDDEN_SIZE,
        NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        label_smoothing=LABEL_SMOOTHING)

    return vae_train, vae_encoder_val, vae_decoder_val, latent_classifier


class GraphVAETrainer(Trainer):
    def train_epoch(self):
        vae = self.model_train[0].train()
        latent_classifier = self.model_train[1].train()

        pbar = tqdm(self.data_loader_train[0], desc="Training Loss: ?")
        for step, data in enumerate(pbar, start=1):
            data = recursive_to_device(data, self.device, non_blocking=True)

            label_vector = data['label_vector']
            graph_encoder_inputs = data['graph_encoder_inputs']
            graph_decoder_inputs = data['graph_decoder_inputs']

            vae_outputs = vae(
                graph_encoder_inputs,
                graph_decoder_inputs,
                calc_loss=True)

            latent_z = vae_outputs['vae_outputs']['latent_z'].detach()
            latent_classifier_outputs = latent_classifier(
                latent_z,
                labels=label_vector,
                calc_loss=True)

            loss = vae_outputs['vae_outputs']['loss'] \
                + latent_classifier_outputs['loss']
            loss_normalized = loss / self.gradient_accumulation_steps
            loss_normalized.backward()

            if step % self.gradient_accumulation_steps == 0 \
                    or step >= len(self.data_loader_train[0]):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.global_step % self.steps_per_log == 0:
                reconstruction_loss = \
                    vae_outputs['vae_outputs']['reconstruction_loss']
                kl_loss = vae_outputs['vae_outputs']['kl_loss']
                beta = vae_outputs['vae_outputs']['beta']

                self.writer.add_scalar(
                    'loss', loss, global_step=self.global_step)
                self.writer.add_scalar(
                    'reconstruction_loss',
                    reconstruction_loss,
                    global_step=self.global_step)
                self.writer.add_scalar(
                    'kl_loss', kl_loss, global_step=self.global_step)
                self.writer.add_scalar(
                    'beta', beta, global_step=self.global_step)

                pbar.set_description(
                    "Training Loss: {:.3g}".format(loss.item()))

            self.global_step += 1

    def eval(self):
        vae_encoder = self.model_val[0].eval()
        vae_decoder = self.model_val[1].eval()
        latent_classifier = self.model_val[2].eval()

        locs = []
        logscales = []
        preds = []
        labels = []

        pbar = tqdm(self.data_loader_val[0], desc="Evaluation")
        for step, data in enumerate(pbar, start=1):
            data = recursive_to_device(data, self.device, non_blocking=True)

            label_vector = data['label_vector']
            graph_encoder_inputs = data['graph_encoder_inputs']
            graph_decoder_inputs = data['graph_decoder_inputs']

            vae_outputs = vae_encoder(
                graph_encoder_inputs,
                graph_decoder_inputs,
                calc_loss=False)

            latent_z = vae_outputs['vae_outputs']['latent_z']
            latent_classifier_outputs = latent_classifier(latent_z)
            latent_classifier_outputs['outputs'] = torch.round(torch.sigmoid(
                latent_classifier_outputs['outputs']))

            locs.append(vae_outputs['encoder_outputs']['loc'])
            logscales.append(vae_outputs['encoder_outputs']['logscale'])
            preds.append(latent_classifier_outputs['outputs'])
            labels.append(label_vector)
        print()

        locs = torch.cat(locs)
        logscales = torch.cat(logscales)
        preds = torch.cat(preds).detach().cpu().numpy()
        labels = torch.cat(labels).detach().cpu().numpy()

        print(classification_report(labels, preds))
        self.writer.add_scalar(
            'f1_score',
            f1_score(labels, preds, average='macro'),
            global_step=self.epoch)

        ranks_list = []
        pbar = tqdm(self.data_loader_val[1], desc="Evaluation")
        for step, data in enumerate(pbar, start=1):
            data = recursive_to_device(data, self.device, non_blocking=True)

            sentence_index = data['sentence_index']
            triples = data['triples']
            triples_mask = data['triples_mask']

            loc = F.embedding(sentence_index, locs)
            logscale = F.embedding(sentence_index, logscales)

            graph_encoder_outputs = {
                'loc': loc,
                'logscale': logscale,
            }

            graph_decoder_inputs = {
                'triples': triples,
            }

            vae_outputs = vae_decoder(
                graph_encoder_outputs,
                graph_decoder_inputs,
                calc_loss=False)

            triple_scores = vae_outputs['decoder_outputs']['triple_scores']
            triple_scores = triple_scores * triples_mask

            ranks = get_triple_ranks(
                triple_scores, descending=True, method='average')
            ranks_list.append(ranks)

            if step >= self.max_eval_steps:
                break
        print()

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

        self.writer.add_scalar('mean_rank', mean_rank, global_step=self.epoch)
        self.writer.add_scalar(
            'mean_reciprocal_rank',
            mean_reciprocal_rank,
            global_step=self.epoch)
        self.writer.add_scalar('hits_1', hits_1, global_step=self.epoch)
        self.writer.add_scalar('hits_3', hits_3, global_step=self.epoch)
        self.writer.add_scalar('hits_10', hits_10, global_step=self.epoch)

    def save_model(self):
        self.model_train[0].encoder.save_pretrained(KBGAT_ENCODER_SAVE_PATH)
        self.model_train[0].decoder.save_pretrained(CONVE_DECODER_SAVE_PATH)


if __name__ == '__main__':
    main()
