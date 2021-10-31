from collections import abc

import numpy as np
import torch
from torch.utils.data import get_worker_info


def filter_dict(d, keys):
    return {k: d[k] for k in keys}


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    return {k: [d[k] for d in list_of_dicts] for k in list_of_dicts[0].keys()}


def random_worker_init_fn(worker_id):
    worker_info = get_worker_info()
    worker_info.dataset.set_rng(np.random.default_rng(worker_info.seed))


def recursive_to_device(data, device, non_blocking=False):
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, abc.Mapping):
        return {
            key: recursive_to_device(value, device, non_blocking=non_blocking)
            for key, value in data.items()}
    elif isinstance(data, abc.Sequence):
        return [
            recursive_to_device(item, device, non_blocking=non_blocking)
            for item in data]
    else:
        return data
