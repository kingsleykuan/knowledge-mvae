import optuna
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..models import (ConvKB,
                      KBGraphAttentionNetwork,
                      KnowledgeGraphEmbeddings)
from ..triple_data import (GraphTripleTrainDataset,
                           TripleTestDataset,
                           convert_triple_list_to_sparse_triple_adjacency_list,
                           random_worker_init_fn)
from ..triple_eval import get_triple_ranks

CONFIG = {
    'knowledge_graph_train_path': 'data/freebase/train',
    'knowledge_graph_val_path': 'data/freebase/val',
    'knowledge_graph_test_path': 'data/freebase/test',

    'pretrained_embeddings_path': 'models/transe_embeddings.pt',
    'pretrained_kbgat_path': None,
    'pretrained_convkb_path': None,
    'kbgat_path': 'models/kbgat_ae.pt',
    'convkb_path': 'models/convkb_ae.pt',

    'num_epochs': 1000,
    'steps_per_log': 100,
    'epochs_per_val': 10,
    'max_eval_steps': 500,
    'num_threads': 16,

    'batch_size': 1024,
    'test_batch_size': 16,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,

    'embedding_size': 100,
    'num_heads': 2,
    'graph_radius': 2,
    'num_filters': 50,
    'dropout_rate': 0.5,
    'negative_ratio': 20,
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

        config['num_filters'] = trial.suggest_int(
            'num_filters', 10, 50, step=10)
        config['negative_ratio'] = trial.suggest_int(
            'negative_ratio', 10, 30, step=5)
        config['label_smoothing'] = trial.suggest_float(
            'label_smoothing', 0.0, 0.3, step=0.1)

        return train(**config)

    def print_status(study, trial):
        print("Trial Params:\n {}".format(trial.params))
        print("Trial Value:\n {}".format(trial.value))
        print("Best Params:\n {}".format(study.best_params))
        print("Best Value:\n {}".format(study.best_value))

    study = optuna.create_study(
        study_name='kbgat_convkb_ae', direction='maximize')
    study.optimize(objective, n_trials=50, callbacks=[print_status])
    print(study.best_params)
    print(study.best_value)


def train(
        knowledge_graph_train_path,
        knowledge_graph_val_path,
        pretrained_embeddings_path,
        pretrained_kbgat_path,
        pretrained_convkb_path,
        kbgat_path,
        convkb_path,
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
        label_smoothing,
        **kwargs):
    triple_train_dataset, data_loader_train = load_data_train(
        knowledge_graph_train_path,
        batch_size,
        num_threads,
        graph_radius,
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

    triple_list_val = list(triple_train_dataset.graph.edges(keys=True))
    triple_list_val, sparse_triple_adjacency_list_indices_val = \
        convert_triple_list_to_sparse_triple_adjacency_list(
            triple_list_val,
            triple_train_dataset.num_entities,
            add_identity=True,
            identity_relation_id=triple_train_dataset.identity_relation_id)
    triple_list_val = torch.from_numpy(triple_list_val).to(device)
    sparse_triple_adjacency_list_indices_val = torch.from_numpy(
        sparse_triple_adjacency_list_indices_val).to(device)
    entity_ids_val = torch.arange(
        0, triple_train_dataset.num_entities, device=device)
    relation_ids_val = torch.arange(
        0, triple_train_dataset.num_relations, device=device)

    kbgat, convkb = load_models(
        triple_train_dataset,
        embedding_size,
        num_heads,
        num_filters,
        dropout_rate=dropout_rate,
        pretrained_embeddings_path=pretrained_embeddings_path,
        pretrained_kbgat_path=pretrained_kbgat_path,
        pretrained_convkb_path=pretrained_convkb_path)
    kbgat = kbgat.to(device)
    convkb = convkb.to(device)

    parameters = kbgat.parameter_dicts() + [{'params': convkb.parameters()}]
    optimizer = optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay)

    writer = SummaryWriter(
        comment='LR{:.3g}_WD{:.3g}_NF{}_NR{}_LS{}'.format(
            learning_rate,
            weight_decay,
            num_filters,
            negative_ratio,
            label_smoothing))

    global_step = 1
    best_mean_reciprocal_rank = 0.0
    best_epoch = None

    for epoch in range(1, num_epochs + 1):
        loss, global_step = train_loop(
            data_loader_train,
            kbgat,
            convkb,
            optimizer,
            writer,
            global_step,
            epoch,
            steps_per_log,
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
                    entity_ids_val,
                    relation_ids_val,
                    triple_list_val,
                    sparse_triple_adjacency_list_indices_val,
                    kbgat,
                    convkb,
                    writer=writer,
                    epoch=epoch,
                    filtered=True,
                    max_eval_steps=max_eval_steps)

                if mean_reciprocal_rank > best_mean_reciprocal_rank:
                    best_mean_reciprocal_rank = mean_reciprocal_rank
                    best_epoch = epoch
                    save_models(kbgat, convkb, kbgat_path, convkb_path)

    print("Best Epoch: {0:.5g}, Best Mean Reciprocal Rank: {1:.5g}".format(
        best_epoch, best_mean_reciprocal_rank))

    return best_mean_reciprocal_rank


def test(
        knowledge_graph_train_path,
        knowledge_graph_val_path,
        knowledge_graph_test_path,
        kbgat_path,
        convkb_path,
        max_eval_steps,
        num_threads,
        test_batch_size,
        embedding_size,
        num_heads,
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

    triple_train_dataset = GraphTripleTrainDataset(knowledge_graph_train_path)

    triple_list_test = list(triple_train_dataset.graph.edges(keys=True))
    triple_list_test, sparse_triple_adjacency_list_indices_test = \
        convert_triple_list_to_sparse_triple_adjacency_list(
            triple_list_test,
            triple_train_dataset.num_entities,
            add_identity=True,
            identity_relation_id=triple_train_dataset.identity_relation_id)
    triple_list_test = torch.from_numpy(triple_list_test).to(device)
    sparse_triple_adjacency_list_indices_test = torch.from_numpy(
        sparse_triple_adjacency_list_indices_test).to(device)
    entity_ids_test = torch.arange(
        0, triple_train_dataset.num_entities, device=device)
    relation_ids_test = torch.arange(
        0, triple_train_dataset.num_relations, device=device)

    kbgat, convkb = load_models(
        triple_test_dataset,
        embedding_size,
        num_heads,
        num_filters,
        pretrained_kbgat_path=kbgat_path,
        pretrained_convkb_path=convkb_path)
    kbgat = kbgat.to(device)
    convkb = convkb.to(device)

    with torch.no_grad():
        (
            mean_rank,
            mean_reciprocal_rank,
            hits_1,
            hits_3,
            hits_10,
        ) = eval_loop(
            data_loader_test,
            entity_ids_test,
            relation_ids_test,
            triple_list_test,
            sparse_triple_adjacency_list_indices_test,
            kbgat,
            convkb,
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
        embedding_size,
        num_heads,
        num_filters,
        dropout_rate=1.0,
        pretrained_embeddings_path=None,
        pretrained_kbgat_path=None,
        pretrained_convkb_path=None):
    if pretrained_embeddings_path:
        knowledge_graph_embeddings = KnowledgeGraphEmbeddings(
            dataset.num_entities,
            dataset.num_relations,
            embedding_size)
        knowledge_graph_embeddings.load_state_dict(
            torch.load(pretrained_embeddings_path))
        entity_embeddings, relation_embeddings = knowledge_graph_embeddings()
    else:
        entity_embeddings, relation_embeddings = None, None

    kbgat = KBGraphAttentionNetwork(
        dataset.num_entities,
        dataset.num_relations,
        embedding_size,
        num_heads,
        dropout_rate=dropout_rate,
        normalize_embeddings=True,
        pretrained_entity_embeddings=entity_embeddings,
        pretrained_relation_embeddings=relation_embeddings)
    if pretrained_kbgat_path:
        kbgat.load_state_dict(torch.load(pretrained_kbgat_path))

    convkb = ConvKB(
        embedding_size, num_filters, dropout_rate=dropout_rate)
    if pretrained_convkb_path:
        convkb.load_state_dict(torch.load(pretrained_convkb_path))

    return kbgat, convkb


def train_loop(
        data_loader,
        kbgat,
        convkb,
        optimizer,
        writer,
        global_step,
        epoch,
        steps_per_log,
        label_smoothing=0.0):
    kbgat = kbgat.train()
    convkb = convkb.train()

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

        loss = train_step(
            triples,
            corrupted_triples,
            entity_ids,
            relation_ids,
            triple_list,
            sparse_triple_adjacency_list_indices,
            kbgat,
            convkb,
            label_smoothing=label_smoothing)

        loss.backward()
        optimizer.step()

        if global_step % steps_per_log == 0:
            print("Step: {0}, Epoch: {1}, Loss: {2:.5g}".format(
                global_step, epoch, loss))
            writer.add_scalar('loss', loss, global_step=global_step)

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
        kbgat,
        convkb,
        label_smoothing=0.0):
    entity_embeddings, relation_embeddings = kbgat(
        entity_ids,
        relation_ids,
        triple_list,
        sparse_triple_adjacency_list_indices)

    triple_labels = torch.cat((
        torch.full((len(triples),), 1.0 - label_smoothing, device=device),
        torch.full((len(corrupted_triples),), label_smoothing, device=device)))

    combined_triples = torch.cat((triples, corrupted_triples), 0)

    triple_scores = convkb(
        combined_triples, entity_embeddings, relation_embeddings)

    loss = F.binary_cross_entropy_with_logits(triple_scores, triple_labels)
    return loss


def eval_loop(
        data_loader,
        entity_ids,
        relation_ids,
        triple_list,
        sparse_triple_adjacency_list_indices,
        kbgat,
        convkb,
        writer=None,
        epoch=None,
        filtered=False,
        max_eval_steps=float('inf')):
    kbgat = kbgat.eval()
    convkb = convkb.eval()

    entity_embeddings, relation_embeddings = kbgat(
        entity_ids,
        relation_ids,
        triple_list,
        sparse_triple_adjacency_list_indices)

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
            convkb,
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
        convkb,
        mask=None):
    test_triples_score = convkb(
        test_triples, entity_embeddings, relation_embeddings)
    if mask is not None:
        test_triples_score = test_triples_score * mask

    ranks = get_triple_ranks(
        test_triples_score, descending=True, method='average')
    return ranks


def save_models(kbgat, convkb, kbgat_path, convkb_path):
    kbgat = kbgat.eval()
    convkb = convkb.eval()
    torch.save(kbgat.state_dict(), kbgat_path)
    torch.save(convkb.state_dict(), convkb_path)


if __name__ == '__main__':
    # tune_hyperparameters(CONFIG)
    train(**CONFIG)
    test(**CONFIG)
