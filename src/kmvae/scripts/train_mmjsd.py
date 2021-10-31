import torch
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader

from kmvae.data_utils import random_worker_init_fn, recursive_to_device
from kmvae.kgnn.vae_models import ConvEDecoder, KBGraphAttentionNetworkEncoder
from kmvae.mvae.models import (FeatureDecoder,
                               FeatureEncoder,
                               MultimodalVariationalAutoencoderJSD)
from kmvae.optimus.tokenizers import (get_optimus_bert_tokenizer,
                                      get_optimus_gpt2_tokenizer)
from kmvae.optimus.vae_models import OptimusBertEncoder, OptimusGPT2Decoder
from kmvae.sentence_data import SentenceKnowledgeGraphDataset
from kmvae.trainer import Trainer

SENTENCES_TRAIN_PATH = 'data/goemotions/data/train.json'
SENTENCES_VAL_PATH = 'data/goemotions/data/val.json'

KNOWLEDGE_GRAPH_PATH = 'data/emotional-context/emotional-context'


LABEL_ENCODER_PATH = 'models/label_encoder_new'
LABEL_DECODER_PATH = 'models/label_decoder_new'

OPTIMUS_BERT_PRETRAINED_PATH = 'models/optimus/bert_new'
OPTIMUS_GPT2_PRETRAINED_PATH = 'models/optimus/gpt2_new'

KBGAT_ENCODER_PATH = 'models/kbgat_encoder_200_new'
CONVE_DECODER_PATH = 'models/conve_decoder_200_new'


LABEL_ENCODER_SAVE_PATH = 'models/mmjsd_all/label_encoder'
LABEL_DECODER_SAVE_PATH = 'models/mmjsd_all/label_decoder'

OPTIMUS_BERT_SAVE_PATH = 'models/mmjsd_all/bert'
OPTIMUS_GPT2_SAVE_PATH = 'models/mmjsd_all/gpt2'

KBGAT_ENCODER_SAVE_PATH = 'models/mmjsd_all/kbgat_encoder_200'
CONVE_DECODER_SAVE_PATH = 'models/mmjsd_all/conve_decoder_200'


NUM_EPOCHS = 1000
STEPS_PER_LOG = 1000

BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5

EPOCHS_PER_VAL = 1
MAX_EVAL_STEPS = float('inf')

RESET_LATENT = True
NUM_CLASSES = 28
NEGATIVE_RATIO = 10


def main():
    data_loader_train, dataset_train = load_data(
        SENTENCES_TRAIN_PATH, KNOWLEDGE_GRAPH_PATH)
    data_loader_val, dataset_val = load_data(
        SENTENCES_VAL_PATH, KNOWLEDGE_GRAPH_PATH)

    mvae_train, mvae_val = load_model(dataset_train)

    trainer = MVAETrainer(
        [data_loader_train],
        [data_loader_val],
        [mvae_train],
        [mvae_val],
        NUM_EPOCHS,
        STEPS_PER_LOG,
        EPOCHS_PER_VAL,
        MAX_EVAL_STEPS,
        GRADIENT_ACCUMULATION_STEPS,
        LEARNING_RATE,
        WEIGHT_DECAY)

    trainer.train()


def load_data(sentences_path, knowledge_graph_path):
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

    return data_loader, dataset


def load_model(dataset):
    label_encoder = FeatureEncoder.from_pretrained(LABEL_ENCODER_PATH)
    label_decoder = FeatureDecoder.from_pretrained(LABEL_DECODER_PATH)

    bert_encoder = OptimusBertEncoder.from_pretrained(
        OPTIMUS_BERT_PRETRAINED_PATH)
    gpt2_decoder = OptimusGPT2Decoder.from_pretrained(
        OPTIMUS_GPT2_PRETRAINED_PATH)

    kbgat_encoder = KBGraphAttentionNetworkEncoder.from_pretrained(
        KBGAT_ENCODER_PATH)
    conve_decoder = ConvEDecoder.from_pretrained(CONVE_DECODER_PATH)

    if RESET_LATENT:
        label_encoder.reset_latent()
        label_decoder.reset_latent()

        bert_encoder.reset_latent()
        gpt2_decoder.reset_latent()

        kbgat_encoder.reset_latent()
        conve_decoder.reset_latent()

    mvae_train = MultimodalVariationalAutoencoderJSD(
        (
            (label_encoder, label_decoder),
            (bert_encoder, gpt2_decoder),
            (kbgat_encoder, conve_decoder),
        ),
        kl_free_bits=2.0,
        beta_schedule='cyclical',
        beta_start=0.0,
        beta_stop=1.0,
        beta_cyclical_total_steps=2702000,
        beta_cycles=1000,
        beta_cycle_ratio_zero=0.5,
        beta_cycle_ratio_increase=0.25,
        loss_reduction='mean')

    mvae_val = MultimodalVariationalAutoencoderJSD(
        (
            (None, label_decoder),
            (bert_encoder, None),
            (kbgat_encoder, None),
        ))

    return mvae_train, mvae_val


class MVAETrainer(Trainer):
    def train_epoch(self):
        mvae = self.model_train[0].train()

        pbar = tqdm(self.data_loader_train[0], desc="Training Loss: ?")
        for step, data in enumerate(pbar, start=1):
            data = recursive_to_device(data, self.device, non_blocking=True)

            label_vector = data['label_vector'].float()
            label_encoder_inputs = {'features': label_vector}
            label_decoder_inputs = {'labels': label_vector}
            text_encoder_inputs = data['text_encoder_inputs']
            text_decoder_inputs = data['text_decoder_inputs']
            graph_encoder_inputs = data['graph_encoder_inputs']
            graph_decoder_inputs = data['graph_decoder_inputs']

            outputs = mvae(
                (
                    (label_encoder_inputs, label_decoder_inputs),
                    (text_encoder_inputs, text_decoder_inputs),
                    (graph_encoder_inputs, graph_decoder_inputs),
                ),
                calc_loss=True)

            loss = outputs['mvae_outputs']['loss']
            loss_normalized = loss / self.gradient_accumulation_steps
            loss_normalized.backward()

            if step % self.gradient_accumulation_steps == 0 \
                    or step >= len(self.data_loader_train[0]):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.global_step % self.steps_per_log == 0:
                reconstruction_loss = \
                    outputs['mvae_outputs']['reconstruction_loss']
                mmjsd_loss = outputs['mvae_outputs']['mmjsd_loss']
                beta = outputs['mvae_outputs']['beta']

                self.writer.add_scalar(
                    'loss', loss, global_step=self.global_step)
                self.writer.add_scalar(
                    'reconstruction_loss',
                    reconstruction_loss,
                    global_step=self.global_step)
                self.writer.add_scalar(
                    'mmjsd_loss', mmjsd_loss, global_step=self.global_step)
                self.writer.add_scalar(
                    'beta', beta, global_step=self.global_step)

                pbar.set_description(
                    "Training Loss: {:.3g}".format(loss.item()))

            self.global_step += 1

    def eval(self):
        mvae = self.model_val[0].eval()

        preds = []
        labels = []

        pbar = tqdm(self.data_loader_val[0], desc="Evaluation")
        for step, data in enumerate(pbar, start=1):
            data = recursive_to_device(data, self.device, non_blocking=True)

            label_vector = data['label_vector'].float()
            label_decoder_inputs = {'labels': label_vector}
            text_encoder_inputs = data['text_encoder_inputs']
            graph_encoder_inputs = data['graph_encoder_inputs']

            outputs = mvae(
                (
                    (None, label_decoder_inputs),
                    (text_encoder_inputs, None),
                    (graph_encoder_inputs, None),
                ),
                calc_loss=False)

            preds.append(torch.round(torch.sigmoid(
                outputs['decoder_outputs'][0]['features'])))
            labels.append(label_vector)

            if step >= self.max_eval_steps:
                break
        print()

        preds = torch.cat(preds).detach().cpu().numpy()
        labels = torch.cat(labels).detach().cpu().numpy()

        print(classification_report(labels, preds))
        self.writer.add_scalar(
            'f1_score',
            f1_score(labels, preds, average='macro'),
            global_step=self.epoch)

    def save_model(self):
        mvae = self.model_train[0].eval()

        label_encoder = mvae.encoder_decoder_models[0][0]
        label_decoder = mvae.encoder_decoder_models[0][1]

        bert_encoder = mvae.encoder_decoder_models[1][0]
        gpt2_decoder = mvae.encoder_decoder_models[1][1]

        kbgat_encoder = mvae.encoder_decoder_models[2][0]
        conve_decoder = mvae.encoder_decoder_models[2][1]

        label_encoder.save_pretrained(LABEL_ENCODER_SAVE_PATH)
        label_decoder.save_pretrained(LABEL_DECODER_SAVE_PATH)

        bert_encoder.save_pretrained(OPTIMUS_BERT_SAVE_PATH)
        gpt2_decoder.save_pretrained(OPTIMUS_GPT2_SAVE_PATH)

        kbgat_encoder.save_pretrained(KBGAT_ENCODER_SAVE_PATH)
        conve_decoder.save_pretrained(CONVE_DECODER_SAVE_PATH)


if __name__ == '__main__':
    main()
