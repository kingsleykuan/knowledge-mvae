"""TransE Knowledge Graph Embeddings."""
import optuna
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..models import (KnowledgeGraphEmbeddings,
                      TransE)
from ..triple_data import (TripleTestDataset,
                           TripleTrainDataset,
                           random_worker_init_fn)
from ..triple_eval import get_triple_ranks

CONFIG = {
    'knowledge_graph_train_path': 'data/emotional-context/train',
    'knowledge_graph_val_path': 'data/emotional-context/val',
    'knowledge_graph_test_path': 'data/emotional-context/test',

    'transe_embeddings_path': 'models_200/transe_embeddings.pt',

    'num_epochs': 2000,
    'epochs_per_val': 200,
    'max_eval_steps': float('inf'),
    'num_threads': 16,

    'batch_size': 2048,
    'test_batch_size': 64,
    'learning_rate': 1e-3,

    'embedding_size': 200,
    'norm': 1,
    'margin': 2,
    'negative_ratio': 20,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tune_hyperparameters(config):
    def objective(trial):
        config['num_epochs'] = 300

        # config['embedding_size'] = trial.suggest_int(
        #     'embedding_size', 50, 200, step=50)
        config['margin'] = trial.suggest_int('margin', 1, 5)
        config['negative_ratio'] = trial.suggest_int(
            'negative_ratio', 10, 30, step=5)

        return train(**config)

    def print_status(study, trial):
        print("Trial Params:\n {}".format(trial.params))
        print("Trial Value:\n {}".format(trial.value))
        print("Best Params:\n {}".format(study.best_params))
        print("Best Value:\n {}".format(study.best_value))

    study = optuna.create_study(study_name='transe', direction='maximize')
    study.optimize(objective, n_trials=50, callbacks=[print_status])
    print(study.best_params)
    print(study.best_value)


def train(
        knowledge_graph_train_path,
        knowledge_graph_val_path,
        transe_embeddings_path,
        num_epochs,
        epochs_per_val,
        max_eval_steps,
        num_threads,
        batch_size,
        test_batch_size,
        learning_rate,
        embedding_size,
        norm,
        margin,
        negative_ratio,
        **kwargs):
    triple_train_dataset, data_loader_train = load_data_train(
        knowledge_graph_train_path, batch_size, num_threads, negative_ratio)
    triple_val_dataset, data_loader_val = load_data_test(
        knowledge_graph_val_path,
        test_batch_size,
        num_threads,
        filtered=True,
        all_knowledge_graph_paths=[
            knowledge_graph_train_path,
            knowledge_graph_val_path,
        ])

    knowledge_graph_embeddings, transe = load_models(
        triple_train_dataset, embedding_size, norm)
    knowledge_graph_embeddings = knowledge_graph_embeddings.to(device)
    transe = transe.to(device)

    parameters = (
        list(knowledge_graph_embeddings.parameters())
        + list(transe.parameters()))
    optimizer = optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)

    writer = SummaryWriter(
        comment='ES{}_M{}_NR{}'.format(embedding_size, margin, negative_ratio))

    best_mean_reciprocal_rank = 0.0
    best_epoch = None

    for epoch in range(1, num_epochs + 1):
        train_loop(
            data_loader_train,
            knowledge_graph_embeddings,
            transe,
            margin,
            optimizer,
            writer,
            epoch)

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
                    transe,
                    writer=writer,
                    epoch=epoch,
                    filtered=True,
                    max_eval_steps=max_eval_steps)

                if mean_reciprocal_rank > best_mean_reciprocal_rank:
                    best_mean_reciprocal_rank = mean_reciprocal_rank
                    best_epoch = epoch
                    save_models(
                        knowledge_graph_embeddings, transe_embeddings_path)

    print("Best Epoch: {0:.5g}, Best Mean Reciprocal Rank: {1:.5g}".format(
        best_epoch, best_mean_reciprocal_rank))

    return best_mean_reciprocal_rank


def test(
        knowledge_graph_train_path,
        knowledge_graph_val_path,
        knowledge_graph_test_path,
        transe_embeddings_path,
        max_eval_steps,
        num_threads,
        test_batch_size,
        embedding_size,
        norm,
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

    knowledge_graph_embeddings, transe = load_models(
        triple_test_dataset, embedding_size, norm)

    knowledge_graph_embeddings.load_state_dict(
        torch.load(transe_embeddings_path))
    knowledge_graph_embeddings = knowledge_graph_embeddings.to(device)

    transe = transe.to(device)

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
            transe,
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


def load_models(dataset, embedding_size, norm):
    knowledge_graph_embeddings = KnowledgeGraphEmbeddings(
        dataset.num_entities,
        dataset.num_relations,
        embedding_size)

    transe = TransE(norm=norm)

    return knowledge_graph_embeddings, transe


def train_loop(
        data_loader,
        knowledge_graph_embeddings,
        transe,
        margin,
        optimizer,
        writer,
        epoch):
    knowledge_graph_embeddings = knowledge_graph_embeddings.train()
    transe = transe.train()

    for data in data_loader:
        data = [item.to(device, non_blocking=True) for item in data]
        triples, corrupted_triples = data

        optimizer.zero_grad(set_to_none=True)

        loss = train_step(
            triples,
            corrupted_triples,
            knowledge_graph_embeddings,
            transe,
            margin)

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
        transe,
        margin):
    entity_embeddings, relation_embeddings = \
        knowledge_graph_embeddings()

    triples_distance = transe(
        triples, entity_embeddings, relation_embeddings)
    corrupted_triples_distance = transe(
        corrupted_triples, entity_embeddings, relation_embeddings)

    triples_distance = triples_distance.repeat(
        int(len(corrupted_triples_distance) / len(triples_distance)))

    loss = F.margin_ranking_loss(
        triples_distance,
        corrupted_triples_distance,
        -1 * torch.ones_like(triples_distance),
        margin=margin,
        reduction='mean')

    return loss


def eval_loop(
        data_loader,
        knowledge_graph_embeddings,
        transe,
        writer=None,
        epoch=None,
        filtered=False,
        max_eval_steps=float('inf')):
    knowledge_graph_embeddings = knowledge_graph_embeddings.eval()
    transe = transe.eval()

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
            transe,
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
        transe,
        mask=None):
    test_triples_distance = transe(
        test_triples, entity_embeddings, relation_embeddings)
    if mask is not None:
        test_triples_distance = test_triples_distance * mask

    ranks = get_triple_ranks(
        test_triples_distance, descending=False, method='average')
    return ranks


def save_models(knowledge_graph_embeddings, transe_embeddings_path):
    knowledge_graph_embeddings = knowledge_graph_embeddings.eval()
    torch.save(knowledge_graph_embeddings.state_dict(), transe_embeddings_path)


if __name__ == '__main__':
    # tune_hyperparameters(CONFIG)
    # train(**CONFIG)
    test(**CONFIG)
