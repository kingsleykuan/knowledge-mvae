import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..vae_models import (ConvKBDecoder,
                          KBGraphAttentionNetworkEncoder)
from ..triple_data import (GraphTripleTrainDataset,
                           GraphTripleTestDataset,
                           random_worker_init_fn)
from ..triple_eval import get_triple_ranks
from kmvae.mvae.models import VariationalAutoencoder

# NOTE: Currently only 1 context node is used in link-prediction
# evaluation, leading to theoretically lower performance.

CONFIG = {
    'knowledge_graph_train_path': 'data/freebase/train',
    'knowledge_graph_val_path': 'data/freebase/val',
    'knowledge_graph_test_path': 'data/freebase/test',

    'pretrained_kbgat_path': 'models/kbgat_ae.pt',
    'pretrained_convkb_path': 'models/convkb_ae.pt',
    'kbgat_convkb_vae_path': 'models/kbgat_convkb_vae.pt',

    'num_epochs': 1000,
    'steps_per_log': 100,
    'epochs_per_val': 5,
    'max_eval_steps': float('inf'),
    'num_threads': 16,

    'batch_size': 1024,
    'test_batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,

    'embedding_size': 32,
    'num_heads': 2,
    'graph_radius': 2,
    'num_filters': 32,
    'dropout_rate': 0.5,
    'negative_ratio': 2,

    'kl_free_bits': 0.5,
    'beta_schedule': 'cyclical',
    'beta_start': 0.0,
    'beta_stop': 1.0,
    'beta_cyclical_total_steps': 20000,
    'beta_cycles': 16,
    'beta_cycle_ratio_zero': 0.5,
    'beta_cycle_ratio_increase': 0.25,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tune_hyperparameters(config):
    def objective(trial):
        config['num_epochs'] = 300
        config['embedding_size'] = trial.suggest_int(
            'embedding_size', 50, 200, step=50)
        config['margin'] = trial.suggest_int('margin', 1, 10)
        config['negative_ratio'] = trial.suggest_int(
            'negative_ratio', 10, 30, step=5)

        return train(**config)

    def print_status(study, trial):
        print("Trial Params:\n {}".format(trial.params))
        print("Trial Value:\n {}".format(trial.value))
        print("Best Params:\n {}".format(study.best_params))
        print("Best Value:\n {}".format(study.best_value))

    study = optuna.create_study(
        study_name='kbgat_convkb_vae', direction='maximize')
    study.optimize(objective, n_trials=50, callbacks=[print_status])
    print(study.best_params)
    print(study.best_value)


def train(
        knowledge_graph_train_path,
        knowledge_graph_val_path,
        pretrained_kbgat_path,
        pretrained_convkb_path,
        kbgat_convkb_vae_path,
        num_epochs,
        steps_per_log,
        epochs_per_val,
        max_eval_steps,
        num_threads,
        batch_size,
        test_batch_size,
        learning_rate,
        weight_decay,
        embedding_size,
        num_heads,
        graph_radius,
        num_filters,
        dropout_rate,
        negative_ratio,
        kl_free_bits,
        beta_schedule,
        beta_start,
        beta_stop,
        beta_cyclical_total_steps,
        beta_cycles,
        beta_cycle_ratio_zero,
        beta_cycle_ratio_increase,
        **kwargs):
    triple_train_dataset, data_loader_train = load_data_train(
        knowledge_graph_train_path,
        batch_size,
        num_threads,
        graph_radius,
        negative_ratio)
    triple_val_dataset, data_loader_val = load_data_test(
        knowledge_graph_val_path,
        knowledge_graph_train_path,
        test_batch_size,
        num_threads,
        graph_radius,
        filtered=False)

    kbgat_convkb_vae = load_models(
        triple_train_dataset,
        embedding_size,
        num_heads,
        num_filters,
        dropout_rate=dropout_rate,
        kl_free_bits=kl_free_bits,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_stop=beta_stop,
        beta_cyclical_total_steps=beta_cyclical_total_steps,
        beta_cycles=beta_cycles,
        beta_cycle_ratio_zero=beta_cycle_ratio_zero,
        beta_cycle_ratio_increase=beta_cycle_ratio_increase,
        pretrained_kbgat_path=pretrained_kbgat_path,
        pretrained_convkb_path=pretrained_convkb_path)
    kbgat_convkb_vae = kbgat_convkb_vae.to(device)

    parameters = kbgat_convkb_vae.parameter_dicts()
    optimizer = optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay)

    writer = SummaryWriter()

    global_step = 1
    best_mean_reciprocal_rank = 0.0
    best_epoch = None

    for epoch in range(1, num_epochs + 1):
        loss, global_step = train_loop(
            data_loader_train,
            kbgat_convkb_vae,
            optimizer,
            writer,
            global_step,
            epoch,
            steps_per_log)

        if epoch % epochs_per_val == 0:
            with torch.no_grad():
                (
                    mean_rank,
                    mean_reciprocal_rank,
                    hits_1,
                    hits_3,
                    hits_10,
                ) = eval_loop(
                    data_loader_val,
                    kbgat_convkb_vae,
                    writer=writer,
                    epoch=epoch,
                    filtered=False,
                    max_eval_steps=max_eval_steps)

                if mean_reciprocal_rank > best_mean_reciprocal_rank:
                    best_mean_reciprocal_rank = mean_reciprocal_rank
                    best_epoch = epoch
                    save_models(kbgat_convkb_vae, kbgat_convkb_vae_path)

    print("Best Epoch: {0:.5g}, Best Mean Reciprocal Rank: {1:.5g}".format(
        best_epoch, best_mean_reciprocal_rank))

    return best_mean_reciprocal_rank


def test(
        knowledge_graph_train_path,
        knowledge_graph_val_path,
        knowledge_graph_test_path,
        kbgat_convkb_vae_path,
        max_eval_steps,
        num_threads,
        test_batch_size,
        embedding_size,
        num_heads,
        graph_radius,
        num_filters,
        **kwargs):
    triple_test_dataset, data_loader_test = load_data_test(
        knowledge_graph_test_path,
        knowledge_graph_train_path,
        test_batch_size,
        num_threads,
        graph_radius,
        filtered=True,
        all_knowledge_graph_paths=[
            knowledge_graph_train_path,
            knowledge_graph_val_path,
            knowledge_graph_test_path,
        ])

    kbgat_convkb_vae = load_models(
        triple_test_dataset,
        embedding_size,
        num_heads,
        num_filters)

    kbgat_convkb_vae.load_state_dict(torch.load(kbgat_convkb_vae_path))
    kbgat_convkb_vae = kbgat_convkb_vae.to(device)

    with torch.no_grad():
        (
            mean_rank,
            mean_reciprocal_rank,
            hits_1,
            hits_3,
            hits_10,
        ) = eval_loop(
            data_loader_test,
            kbgat_convkb_vae,
            writer=None,
            epoch=None,
            filtered=True,
            max_eval_steps=max_eval_steps)

    return mean_reciprocal_rank


def load_data_train(
        knowledge_graph_path,
        batch_size,
        num_threads,
        graph_radius,
        negative_ratio):
    dataset = GraphTripleTrainDataset(
        knowledge_graph_path,
        radius=graph_radius,
        negative_ratio=negative_ratio,
        filter_triples=True,
        bernoulli_trick=True,
        add_context=True)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_threads,
        collate_fn=dataset.get_collate_fn(),
        pin_memory=True,
        worker_init_fn=random_worker_init_fn,
        persistent_workers=True)

    return dataset, data_loader


def load_data_test(
        knowledge_graph_test_path,
        knowledge_graph_train_path,
        test_batch_size,
        num_threads,
        graph_radius,
        filtered=False,
        all_knowledge_graph_paths=None):
    dataset = GraphTripleTestDataset(
        knowledge_graph_test_path,
        knowledge_graph_train_path,
        radius=graph_radius,
        add_context=True,
        filtered=filtered,
        all_knowledge_graph_paths=all_knowledge_graph_paths)
    data_loader = DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_threads,
        collate_fn=dataset.get_collate_fn(),
        pin_memory=True,
        persistent_workers=True)

    return dataset, data_loader


def load_models(
        dataset,
        embedding_size,
        num_heads,
        num_filters,
        dropout_rate=1.0,
        kl_free_bits=1.0,
        beta_schedule='cyclical',
        beta_start=0.0,
        beta_stop=1.0,
        beta_cyclical_total_steps=1000000,
        beta_cycles=10,
        beta_cycle_ratio_zero=0.5,
        beta_cycle_ratio_increase=0.25,
        pretrained_kbgat_path=None,
        pretrained_convkb_path=None):
    if pretrained_kbgat_path:
        kbgat_state_dict = torch.load(pretrained_kbgat_path)
    else:
        kbgat_state_dict = None

    kbgat = KBGraphAttentionNetworkEncoder(
        dataset.num_entities,
        dataset.num_relations,
        embedding_size,
        num_heads,
        embedding_size,
        dropout_rate=dropout_rate,
        normalize_embeddings=False,
        kbgat_state_dict=kbgat_state_dict)

    if pretrained_convkb_path:
        convkb_state_dict = torch.load(pretrained_convkb_path)
    else:
        convkb_state_dict = None

    convkb = ConvKBDecoder(
        embedding_size,
        num_filters,
        embedding_size,
        dropout_rate=dropout_rate,
        convkb_state_dict=convkb_state_dict)

    kbgat_convkb_vae = VariationalAutoencoder(
        kbgat,
        convkb,
        kl_free_bits=kl_free_bits,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_stop=beta_stop,
        beta_cyclical_total_steps=beta_cyclical_total_steps,
        beta_cycles=beta_cycles,
        beta_cycle_ratio_zero=beta_cycle_ratio_zero,
        beta_cycle_ratio_increase=beta_cycle_ratio_increase,
        loss_reduction='none')

    return kbgat_convkb_vae


def train_loop(
        data_loader,
        kbgat_convkb_vae,
        optimizer,
        writer,
        global_step,
        epoch,
        steps_per_log):
    kbgat_convkb_vae = kbgat_convkb_vae.train()

    for data in data_loader:
        data = [item.to(device, non_blocking=True) for item in data]
        (
            triples,
            corrupted_triples,
            entity_ids,
            relation_ids,
            triple_list,
            sparse_triple_adjacency_list_indices
        ) = data

        optimizer.zero_grad(set_to_none=True)

        loss, outputs = train_step(
            triples,
            corrupted_triples,
            entity_ids,
            relation_ids,
            triple_list,
            sparse_triple_adjacency_list_indices,
            kbgat_convkb_vae)

        loss.backward()
        optimizer.step()

        if global_step % steps_per_log == 0:
            print("Step: {0}, Epoch: {1}, Loss: {2:.5g}".format(
                global_step, epoch, loss))
            writer.add_scalar('loss', loss, global_step=global_step)

            reconstruction_loss = \
                outputs['vae_outputs']['reconstruction_loss']
            kl_loss = outputs['vae_outputs']['kl_loss']
            beta = outputs['vae_outputs']['beta']

            writer.add_scalar(
                'reconstruction_loss',
                reconstruction_loss,
                global_step=global_step)
            writer.add_scalar('kl_loss', kl_loss, global_step=global_step)
            writer.add_scalar('beta', beta, global_step=global_step)

        global_step += 1
    optimizer.zero_grad(set_to_none=True)

    print("Epoch: {0}, Loss: {1:.5g}".format(epoch, loss))
    writer.add_scalar('epoch_loss', loss, global_step=epoch)

    return loss, global_step


def train_step(
        triples,
        corrupted_triples,
        entity_ids,
        relation_ids,
        triple_list,
        sparse_triple_adjacency_list_indices,
        kbgat_convkb_vae):
    triple_labels = torch.cat((
        torch.zeros(len(triples), device=device),
        torch.ones(len(corrupted_triples), device=device)), 0)
    combined_triples = torch.cat((triples, corrupted_triples), 0)

    encoder_inputs = {
        'entity_ids': entity_ids,
        'relation_ids': relation_ids,
        'triple_list': triple_list,
        'sparse_triple_adjacency_list_indices':
            sparse_triple_adjacency_list_indices,
        'context_entity_index': -1,
    }

    decoder_inputs = {
        'triples': combined_triples,
        'triple_labels': triple_labels,
    }

    outputs = kbgat_convkb_vae(encoder_inputs, decoder_inputs, calc_loss=True)
    loss = outputs['vae_outputs']['loss']

    return loss, outputs


def eval_loop(
        data_loader,
        kbgat_convkb_vae,
        writer=None,
        epoch=None,
        filtered=False,
        max_eval_steps=float('inf')):
    kbgat_convkb_vae = kbgat_convkb_vae.eval()

    ranks_list = []
    for i, data in enumerate(data_loader):
        data = [item.to(device, non_blocking=True) for item in data]

        if filtered:
            (
                test_triples,
                mask,
                entity_ids,
                relation_ids,
                triple_list,
                sparse_triple_adjacency_list_indices
            ) = data
        else:
            (
                test_triples,
                entity_ids,
                relation_ids,
                triple_list,
                sparse_triple_adjacency_list_indices
            ) = data
            mask = None

        ranks = eval_step(
            test_triples,
            entity_ids,
            relation_ids,
            triple_list,
            sparse_triple_adjacency_list_indices,
            kbgat_convkb_vae,
            mask=mask)
        ranks_list.append(ranks)

        print(i, end='\r')
        if i >= max_eval_steps:
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

    if writer is not None and epoch is not None:
        writer.add_scalar('mean_rank', mean_rank, global_step=epoch)
        writer.add_scalar(
            'mean_reciprocal_rank', mean_reciprocal_rank, global_step=epoch)
        writer.add_scalar('hits_1', hits_1, global_step=epoch)
        writer.add_scalar('hits_3', hits_3, global_step=epoch)
        writer.add_scalar('hits_10', hits_10, global_step=epoch)

    return (
        mean_rank,
        mean_reciprocal_rank,
        hits_1,
        hits_3,
        hits_10,
    )


def eval_step(
        test_triples,
        entity_ids,
        relation_ids,
        triple_list,
        sparse_triple_adjacency_list_indices,
        kbgat_convkb_vae,
        mask=None):
    encoder_inputs = {
        'entity_ids': entity_ids,
        'relation_ids': relation_ids,
        'triple_list': triple_list,
        'sparse_triple_adjacency_list_indices':
            sparse_triple_adjacency_list_indices,
        'context_entity_index': -1,
    }

    decoder_inputs = {
        'triples': test_triples,
    }

    outputs = kbgat_convkb_vae(encoder_inputs, decoder_inputs)
    test_triples_score = outputs['decoder_outputs']['triple_scores']
    if mask is not None:
        test_triples_score = test_triples_score * mask

    ranks = get_triple_ranks(
        test_triples_score, descending=True, method='average')
    return ranks


def save_models(kbgat_convkb_vae, kbgat_convkb_vae_path):
    kbgat_convkb_vae = kbgat_convkb_vae.eval()
    torch.save(kbgat_convkb_vae.state_dict(), kbgat_convkb_vae_path)


if __name__ == '__main__':
    tune_hyperparameters(CONFIG)
    train(**CONFIG)
    test(**CONFIG)
