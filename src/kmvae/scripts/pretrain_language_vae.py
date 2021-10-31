import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

from kmvae.data_utils import recursive_to_device
from kmvae.mvae.models import LatentClassifier, VariationalAutoencoder
from kmvae.optimus.tokenizers import (get_optimus_bert_tokenizer,
                                      get_optimus_gpt2_tokenizer)
from kmvae.optimus.vae_models import OptimusBertEncoder, OptimusGPT2Decoder
from kmvae.sentence_data import SentenceDataset
from kmvae.trainer import Trainer

SENTENCES_TRAIN_PATH = 'data/goemotions/data/train.json'
SENTENCES_VAL_PATH = 'data/goemotions/data/val.json'


OPTIMUS_BERT_PRETRAINED_PATH = 'models/optimus/bert_new'
OPTIMUS_GPT2_PRETRAINED_PATH = 'models/optimus/gpt2_new'

OPTIMUS_BERT_SAVE_PATH = 'models/optimus/bert_latent'
OPTIMUS_GPT2_SAVE_PATH = 'models/optimus/gpt2_latent'


NUM_EPOCHS = 1000
STEPS_PER_LOG = 1000
EPOCHS_PER_VAL = 1
MAX_EVAL_STEPS = float('inf')

BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5

NUM_CLASSES = 28

LATENT_SIZE = 512
HIDDEN_SIZE = 1024
DROPOUT_RATE = 0.1
LABEL_SMOOTHING = 0.1


def main():
    data_loader_train = load_data(SENTENCES_TRAIN_PATH)
    data_loader_val = load_data(SENTENCES_VAL_PATH)

    vae, latent_classifier = load_model()

    trainer = LanguageVAETrainer(
        [data_loader_train],
        [data_loader_val],
        [vae, latent_classifier],
        [vae, latent_classifier],
        NUM_EPOCHS,
        STEPS_PER_LOG,
        EPOCHS_PER_VAL,
        MAX_EVAL_STEPS,
        GRADIENT_ACCUMULATION_STEPS,
        LEARNING_RATE,
        WEIGHT_DECAY)

    trainer.train()


def load_data(sentences_path):
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
        shuffle=True,
        num_workers=16,
        collate_fn=dataset.get_collate_fn(),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True)

    return data_loader


def load_model():
    bert_encoder = OptimusBertEncoder.from_pretrained(
        OPTIMUS_BERT_PRETRAINED_PATH)
    gpt2_decoder = OptimusGPT2Decoder.from_pretrained(
        OPTIMUS_GPT2_PRETRAINED_PATH)

    vae = VariationalAutoencoder(
        bert_encoder,
        gpt2_decoder,
        kl_free_bits=2.0,
        beta_schedule='cyclical',
        beta_start=0.0,
        beta_stop=1.0,
        beta_cyclical_total_steps=2702000,
        beta_cycles=100,
        beta_cycle_ratio_zero=0.5,
        beta_cycle_ratio_increase=0.25,
        loss_reduction='mean')

    latent_classifier = LatentClassifier(
        LATENT_SIZE,
        HIDDEN_SIZE,
        NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        label_smoothing=LABEL_SMOOTHING)

    return vae, latent_classifier


class LanguageVAETrainer(Trainer):
    def train_epoch(self):
        vae = self.model_train[0].train()
        latent_classifier = self.model_train[1].train()

        pbar = tqdm(self.data_loader_train[0], desc="Training Loss: ?")
        for step, data in enumerate(pbar, start=1):
            data = recursive_to_device(data, self.device, non_blocking=True)

            label_vector = data['label_vector']
            text_encoder_inputs = data['text_encoder_inputs']
            text_decoder_inputs = data['text_decoder_inputs']

            vae_outputs = vae(
                text_encoder_inputs, text_decoder_inputs, calc_loss=True)

            latent_z = vae_outputs['vae_outputs']['latent_z'].detach()
            latent_classifier_outputs = latent_classifier(
                latent_z, labels=label_vector, calc_loss=True)

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
        vae = self.model_val[0].eval()
        latent_classifier = self.model_val[1].eval()

        cross_entropy = []
        preds = []
        labels = []

        pbar = tqdm(self.data_loader_val[0], desc="Evaluation")
        for step, data in enumerate(pbar, start=1):
            data = recursive_to_device(data, self.device, non_blocking=True)

            label_vector = data['label_vector']
            text_encoder_inputs = data['text_encoder_inputs']
            text_decoder_inputs = data['text_decoder_inputs']

            vae_outputs = vae(
                text_encoder_inputs, text_decoder_inputs, calc_loss=True)

            cross_entropy.append(vae_outputs['decoder_outputs']['loss'])

            latent_z = vae_outputs['vae_outputs']['latent_z'].detach()
            latent_classifier_outputs = latent_classifier(
                latent_z, calc_loss=False)
            latent_classifier_outputs['outputs'] = torch.round(torch.sigmoid(
                latent_classifier_outputs['outputs']))

            preds.append(latent_classifier_outputs['outputs'])
            labels.append(label_vector)

            if step >= self.max_eval_steps:
                break
        print()

        cross_entropy = torch.cat(cross_entropy)
        preds = torch.cat(preds).detach().cpu().numpy()
        labels = torch.cat(labels).detach().cpu().numpy()

        perplexity = torch.mean(torch.exp(cross_entropy))
        print("Perplexity: {}".format(perplexity.item()))
        self.writer.add_scalar(
            'perplexity', perplexity, global_step=self.epoch)

        print(classification_report(labels, preds))
        self.writer.add_scalar(
            'f1_score',
            f1_score(labels, preds, average='macro'),
            global_step=self.epoch)

    def save_model(self):
        vae = self.model_train[0].eval()

        bert_encoder = vae.encoder
        gpt2_decoder = vae.decoder

        bert_encoder.save_pretrained(OPTIMUS_BERT_SAVE_PATH)
        gpt2_decoder.save_pretrained(OPTIMUS_GPT2_SAVE_PATH)


if __name__ == '__main__':
    main()
