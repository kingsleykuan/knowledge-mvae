"""Conve Knowledge Graph Embeddings."""
import optuna
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..models import (ConvE,
                      KnowledgeGraphEmbeddings)
from ..triple_data import (TripleTestDataset,
                           TripleTrainDataset,
                           random_worker_init_fn)
from ..triple_eval import get_triple_ranks

# Mean Rank: 251.61, Mean Reciprocal Rank: 0.26944
# Hits@1: 0.1868, Hits@3: 0.29737, Hits@10: 0.43299
CONFIG = {
    'knowledge_graph_train_path': 'data/freebase/train',
    'knowledge_graph_val_path': 'data/freebase/val',
    'knowledge_graph_test_path': 'data/freebase/test',

    'pretrained_embeddings_path': 'models/transe_embeddings.pt',
    'pretrained_conve_path': None,
    'conve_embeddings_path': 'models/conve_embeddings.pt',
    'conve_path': 'models/conve.pt',
    'freeze_embeddings': False,

    'num_epochs': 1000,
    'epochs_per_val': 20,
    'max_eval_steps': 500,
    'num_threads': 16,

    'batch_size': 1024,
    'test_batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,

    'embedding_size': 50,
    'embedding_height': 5,
    'num_filters': 32,
    'dropout_rate': 0.1,
    'negative_ratio': 10,
    'label_smoothing': 0.1,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tune_hyperparameters(config):
    def objective(trial):
        config['num_epochs'] = 100
        config['max_eval_steps'] = 500

        # config['learning_rate'] = trial.suggest_float(
        #     'learning_rate', 1e-5, 1e-3, log=True)
        # config['weight_decay'] = trial.suggest_float(
        #     'weight_decay', 1e-5, 1e-3, log=True)

        # config['num_filters'] = trial.suggest_int(
        #     'num_filters', 10, 50, step=10)
        # config['dropout_rate'] = trial.suggest_float(
        #     'dropout_rate', 0.0, 0.5, step=0.1)
        # config['negative_ratio'] = trial.suggest_int(
        #     'negative_ratio', 10, 30, step=5)
        config['label_smoothing'] = trial.suggest_float(
            'label_smoothing', 0.1, 0.3, step=0.1)

        return train(**config)

    def print_status(study, trial):
        print("Trial Params:\n {}".format(trial.params))
        print("Trial Value:\n {}".format(trial.value))
        print("Best Params:\n {}".format(study.best_params))
        print("Best Value:\n {}".format(study.best_value))

    study = optuna.create_study(study_name='conve', direction='maximize')
    study.optimize(objective, n_trials=100, callbacks=[print_status])
    print(study.best_params)
    print(study.best_value)


def train(
        knowledge_graph_train_path,
        knowledge_graph_val_path,
        pretrained_embeddings_path,
        pretrained_conve_path,
        conve_embeddings_path,
        conve_path,
        freeze_embeddings,
        num_epochs,
        epochs_per_val,
        max_eval_steps,
        num_threads,
        batch_size,
        test_batch_size,
        learning_rate,
        weight_decay,
        embedding_size,
        embedding_height,
        num_filters,
        dropout_rate,
        negative_ratio,
        label_smoothing,
        **kwargs):
    triple_train_dataset, data_loader_train = load_data_train(
        knowledge_graph_train_path,
        batch_size,
        num_threads,
        negative_ratio)
    triple_val_dataset, data_loader_val = load_data_test(
        knowledge_graph_val_path,
        test_batch_size,
        num_threads,
        filtered=True,
        all_knowledge_graph_paths=[
            knowledge_graph_train_path,
            knowledge_graph_val_path,
        ])

    knowledge_graph_embeddings, conve = load_models(
        triple_train_dataset,
        freeze_embeddings,
        embedding_size,
        embedding_height,
        num_filters,
        dropout_rate=dropout_rate,
        pretrained_embeddings_path=pretrained_embeddings_path,
        pretrained_conve_path=pretrained_conve_path)
    knowledge_graph_embeddings = knowledge_graph_embeddings.to(device)
    conve = conve.to(device)

    parameters = \
        knowledge_graph_embeddings.parameter_dicts() + conve.parameter_dicts()
    optimizer = optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay)

    writer = SummaryWriter(
        comment='LR{:.3g}_WD{:.3g}_NF{}_DR{}_NR{}_LS{}'.format(
            learning_rate,
            weight_decay,
            num_filters,
            dropout_rate,
            negative_ratio,
            label_smoothing))

    best_mean_reciprocal_rank = 0.0
    best_epoch = None

    for epoch in range(1, num_epochs + 1):
        train_loop(
            data_loader_train,
            knowledge_graph_embeddings,
            conve,
            optimizer,
            writer,
            epoch,
            label_smoothing=label_smoothing)

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
                    knowledge_graph_embeddings,
                    conve,
                    writer=writer,
                    epoch=epoch,
                    filtered=True,
                    max_eval_steps=max_eval_steps)

                if mean_reciprocal_rank > best_mean_reciprocal_rank:
                    best_mean_reciprocal_rank = mean_reciprocal_rank
                    best_epoch = epoch
                    save_models(
                        knowledge_graph_embeddings,
                        conve,
                        conve_embeddings_path,
                        conve_path)

    print("Best Epoch: {0:.5g}, Best Mean Reciprocal Rank: {1:.5g}".format(
        best_epoch, best_mean_reciprocal_rank))

    return best_mean_reciprocal_rank


def test(
        knowledge_graph_train_path,
        knowledge_graph_val_path,
        knowledge_graph_test_path,
        conve_embeddings_path,
        conve_path,
        max_eval_steps,
        num_threads,
        test_batch_size,
        embedding_size,
        embedding_height,
        num_filters,
        **kwargs):
    triple_test_dataset, data_loader_test = load_data_test(
        knowledge_graph_test_path,
        test_batch_size,
        num_threads,
        filtered=True,
        all_knowledge_graph_paths=[
            knowledge_graph_train_path,
            knowledge_graph_val_path,
            knowledge_graph_test_path,
        ])

    knowledge_graph_embeddings, conve = load_models(
        triple_test_dataset,
        True,
        embedding_size,
        embedding_height,
        num_filters,
        pretrained_embeddings_path=conve_embeddings_path,
        pretrained_conve_path=conve_path)
    knowledge_graph_embeddings = knowledge_graph_embeddings.to(device)
    conve = conve.to(device)

    with torch.no_grad():
        (
            mean_rank,
            mean_reciprocal_rank,
            hits_1,
            hits_3,
            hits_10,
        ) = eval_loop(
            data_loader_test,
            knowledge_graph_embeddings,
            conve,
            writer=None,
            epoch=None,
            filtered=True,
            max_eval_steps=max_eval_steps)

    return mean_reciprocal_rank


def load_data_train(
        knowledge_graph_path,
        batch_size,
        num_threads,
        negative_ratio):
    dataset = TripleTrainDataset(
        knowledge_graph_path,
        negative_ratio=negative_ratio,
        filter_triples=True,
        bernoulli_trick=True)
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
        knowledge_graph_path,
        test_batch_size,
        num_threads,
        filtered=False,
        all_knowledge_graph_paths=None):
    dataset = TripleTestDataset(
        knowledge_graph_path,
        filtered=filtered,
        all_knowledge_graph_paths=all_knowledge_graph_paths)
    data_loader = DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_threads,
        pin_memory=True,
        persistent_workers=True)

    return dataset, data_loader


def load_models(
        dataset,
        freeze_embeddings,
        embedding_size,
        embedding_height,
        num_filters,
        dropout_rate=1.0,
        pretrained_embeddings_path=None,
        pretrained_conve_path=None):
    knowledge_graph_embeddings = KnowledgeGraphEmbeddings(
        dataset.num_entities,
        dataset.num_relations,
        embedding_size)
    if pretrained_embeddings_path:
        knowledge_graph_embeddings.load_state_dict(
            torch.load(pretrained_embeddings_path))
    if freeze_embeddings:
        knowledge_graph_embeddings.requires_grad_(False)

    conve = ConvE(
        embedding_size,
        embedding_height,
        num_filters,
        dropout_rate=dropout_rate)
    if pretrained_conve_path:
        conve.load_state_dict(torch.load(pretrained_conve_path))

    return knowledge_graph_embeddings, conve


def train_loop(
        data_loader,
        knowledge_graph_embeddings,
        conve,
        optimizer,
        writer,
        epoch,
        label_smoothing=0.0):
    knowledge_graph_embeddings = knowledge_graph_embeddings.train()
    conve = conve.train()

    for data in data_loader:
        data = [item.to(device, non_blocking=True) for item in data]
        triples, corrupted_triples = data

        optimizer.zero_grad(set_to_none=True)

        loss = train_step(
            triples,
            corrupted_triples,
            knowledge_graph_embeddings,
            conve,
            label_smoothing=label_smoothing)

        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    print("Epoch: {0}, Loss: {1:.5g}".format(epoch, loss))
    writer.add_scalar('loss', loss, global_step=epoch)

    return loss


def train_step(
        triples,
        corrupted_triples,
        knowledge_graph_embeddings,
        conve,
        label_smoothing=0.0):
    triple_labels = torch.cat((
        torch.full((len(triples),), 1.0 - label_smoothing, device=device),
        torch.full((len(corrupted_triples),), label_smoothing, device=device)))

    triples = torch.cat((triples, corrupted_triples), 0)

    entity_embeddings, relation_embeddings = \
        knowledge_graph_embeddings()

    triple_scores = conve(
        triples, entity_embeddings, relation_embeddings)

    loss = F.binary_cross_entropy_with_logits(triple_scores, triple_labels)
    return loss


def eval_loop(
        data_loader,
        knowledge_graph_embeddings,
        conve,
        writer=None,
        epoch=None,
        filtered=False,
        max_eval_steps=float('inf')):
    knowledge_graph_embeddings = knowledge_graph_embeddings.eval()
    conve = conve.eval()

    entity_embeddings, relation_embeddings = \
        knowledge_graph_embeddings()

    ranks_list = []
    for i, data in enumerate(data_loader):
        if filtered:
            test_triples, mask = data
            mask = mask.to(device, non_blocking=True)
        else:
            test_triples = data
            mask = None

        test_triples = test_triples.to(device, non_blocking=True)

        ranks = eval_step(
            test_triples,
            entity_embeddings,
            relation_embeddings,
            conve,
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
        entity_embeddings,
        relation_embeddings,
        conve,
        mask=None):
    test_triples_scores = conve(
        test_triples, entity_embeddings, relation_embeddings)
    if mask is not None:
        test_triples_scores = test_triples_scores * mask

    ranks = get_triple_ranks(
        test_triples_scores, descending=True, method='average')
    return ranks


def save_models(
        knowledge_graph_embeddings,
        conve,
        conve_embeddings_path,
        conve_path):
    knowledge_graph_embeddings = knowledge_graph_embeddings.eval()
    conve = conve.eval()
    torch.save(knowledge_graph_embeddings.state_dict(), conve_embeddings_path)
    torch.save(conve.state_dict(), conve_path)


if __name__ == '__main__':
    # tune_hyperparameters(CONFIG)
    train(**CONFIG)
    test(**CONFIG)
