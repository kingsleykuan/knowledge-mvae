import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from kmvae.mvae.models import (FeatureDecoder,
                               FeatureEncoder,
                               VariationalAutoencoder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Train Label VAE")

    parser.add_argument(
        '--num_labels', default=28, type=int,
        help='Number of labels.')

    parser.add_argument(
        '--hidden_size', default=1024, type=int,
        help='Hidden size of models.')

    parser.add_argument(
        '--latent_size', default=100, type=int,
        help='Latent size of VAE.')

    parser.add_argument(
        '--dropout_rate', default=0.1, type=float,
        help='Dropout rate.')

    parser.add_argument(
        '--label_smoothing', default=0.1, type=float,
        help='Label Smoothing.')

    parser.add_argument(
        '--kl_free_bits', default=0.2, type=float,
        help='KL divergence free bits threshold.')

    parser.add_argument(
        '--learning_rate', default=1e-3, type=float,
        help='Learning rate.')

    parser.add_argument(
        '--weight_decay', default=0, type=float,
        help='Weight decay.')

    parser.add_argument(
        '--num_steps', default=1000, type=int,
        help='Number of steps to train for.')

    parser.add_argument(
        '--batch_size', default=1024, type=int,
        help='Batch size during training.')

    parser.add_argument(
        '--eval_size', default=4096, type=int,
        help='Number of evaluation samples.')

    parser.add_argument(
        '--steps_per_log', default=100, type=int,
        help='Steps between writing to log.')

    parser.add_argument(
        '--label_encoder_path', default=None, type=str,
        help='Path to save label encoder to (optional).')

    parser.add_argument(
        '--label_decoder_path', default=None, type=str,
        help='Path to save label decoder to (optional).')

    return parser


def pretrain_label_vae(
        num_labels=28,
        hidden_size=1024,
        latent_size=100,
        dropout_rate=0.1,
        label_smoothing=0.1,
        kl_free_bits=0.2,
        learning_rate=1e-3,
        weight_decay=0,
        num_steps=1000,
        batch_size=1024,
        eval_size=4096,
        steps_per_log=100,
        label_encoder_path=None,
        label_decoder_path=None):
    label_encoder = FeatureEncoder(
        num_labels,
        hidden_size,
        latent_size,
        use_embeddings=False,
        dropout_rate=dropout_rate)
    label_decoder = FeatureDecoder(
        num_labels,
        hidden_size,
        latent_size,
        dropout_rate=dropout_rate,
        label_smoothing=label_smoothing,
        distribution_type='bernoulli')

    vae = VariationalAutoencoder(
        label_encoder,
        label_decoder,
        kl_free_bits=kl_free_bits).to(device)

    parameter_dicts = vae.parameter_dicts()
    optimizer = optim.AdamW(
        parameter_dicts, lr=learning_rate, weight_decay=weight_decay)
    writer = SummaryWriter()

    train(
        vae,
        num_labels,
        optimizer,
        num_steps,
        batch_size,
        writer,
        steps_per_log)

    with torch.no_grad():
        eval(vae, num_labels, eval_size, writer)

    if label_encoder_path:
        label_encoder.save_pretrained(label_encoder_path)
    if label_decoder_path:
        label_decoder.save_pretrained(label_decoder_path)


def train(
        vae,
        num_labels,
        optimizer,
        num_steps,
        batch_size,
        writer,
        steps_per_log):
    vae = vae.train()

    pbar = tqdm(range(1, num_steps + 1), desc="Loss: ")
    for step in pbar:
        label_vector = torch.randint(
            0, 2, (batch_size, num_labels), device=device).float()

        label_encoder_inputs = {'features': label_vector}
        label_decoder_inputs = {'labels': label_vector}

        optimizer.zero_grad(set_to_none=True)

        outputs = vae(
            label_encoder_inputs, label_decoder_inputs, calc_loss=True)
        loss = outputs['vae_outputs']['loss']

        loss.backward()
        optimizer.step()

        if step % steps_per_log == 0:
            reconstruction_loss = \
                outputs['vae_outputs']['reconstruction_loss']
            kl_loss = outputs['vae_outputs']['kl_loss']
            beta = outputs['vae_outputs']['beta']

            writer.add_scalar('loss', loss, step)
            writer.add_scalar('reconstruction_loss', reconstruction_loss, step)
            writer.add_scalar('kl_loss', kl_loss, step)
            writer.add_scalar('beta', beta, step)

            pbar.set_description("Loss: {0:.3g}".format(loss.item()))


def eval(vae, num_labels, eval_size, writer):
    vae = vae.eval()

    label_vector = torch.randint(
        0, 2, (eval_size, num_labels), device=device).float()

    label_encoder_inputs = {'features': label_vector}
    label_decoder_inputs = {}

    outputs = vae(label_encoder_inputs, label_decoder_inputs, calc_loss=False)

    latent_z = outputs['vae_outputs']['latent_z']
    posterior_q_z_x = outputs['vae_outputs']['posterior_q_z_x']
    prior_p_z = vae.negative_elbo_loss.prior(device=device)
    outputs = outputs['decoder_outputs']['features']
    preds = torch.round(torch.sigmoid(outputs))

    labels = label_vector.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    print(classification_report(labels, preds))
    writer.add_scalar(
        'f1_score', f1_score(labels, preds, average='macro'))

    log_likelihood = -torch.sum(F.binary_cross_entropy_with_logits(
        outputs, label_vector, reduction='none'), dim=-1)
    log_prior = torch.sum(prior_p_z.log_prob(latent_z), dim=-1)
    log_posterior = torch.sum(posterior_q_z_x.log_prob(latent_z), dim=-1)
    log_marginal_likelihood = \
        torch.logsumexp(log_likelihood + log_prior - log_posterior, dim=-1) \
        - torch.log(torch.tensor(eval_size, device=device))
    print("Log Marginal Likelihood: {}".format(log_marginal_likelihood))


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    pretrain_label_vae(**vars(args))
