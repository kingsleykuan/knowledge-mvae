from abc import ABC, abstractmethod

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer(ABC):
    def __init__(
            self,
            data_loader_train,
            data_loader_val,
            model_train,
            model_val,
            num_epochs,
            steps_per_log,
            epochs_per_val,
            max_eval_steps,
            gradient_accumulation_steps,
            learning_rate,
            weight_decay):
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.model_train = model_train
        self.model_val = model_val

        self.num_epochs = num_epochs
        self.steps_per_log = steps_per_log
        self.epochs_per_val = epochs_per_val
        self.max_eval_steps = max_eval_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_train = [
            model.to(self.device) for model in self.model_train]
        self.model_val = [
            model.to(self.device) for model in self.model_val]

        parameter_dicts = []
        for model in self.model_train:
            parameter_dicts.extend(model.parameter_dicts())
        self.optimizer = optim.AdamW(
            parameter_dicts,
            lr=self.learning_rate,
            weight_decay=self.weight_decay)

        self.writer = SummaryWriter()

        self.epoch = 1
        self.global_step = 1

    def train(self):
        pbar = tqdm(
            range(self.epoch, self.num_epochs + 1),
            desc="Epoch: {}".format(self.epoch))
        for epoch in pbar:
            self.epoch = epoch
            pbar.set_description("Epoch: {}".format(self.epoch))

            self.model_train = [model.train() for model in self.model_train]
            self.train_epoch()

            if epoch % self.epochs_per_val == 0:
                with torch.no_grad():
                    self.model_val = [model.eval() for model in self.model_val]
                    self.eval()

            self.save_model()

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def save_model(self):
        pass
