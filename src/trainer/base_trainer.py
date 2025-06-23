import os
import torch

from abc import abstractmethod
from tqdm.auto import tqdm


class BaseTrainer():
    def __init__(self,
                 model,
                 optimizer,
                 loss_func,
                 config,
                 show_progressbar):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.config = config
        self.show_progressbar = show_progressbar

        self.start_epoch = 0
        self.epochs = config['epochs']
        self.best_loss = None
        self.counter = 0

        self.save_best = config['save_best']
        self.best_model = None

        self.patience = config['patience']
    
    @abstractmethod
    def _train(self):
        raise NotImplementedError
    
    def train(self):
        epoch_iterator = tqdm(range(self.start_epoch, self.epochs), desc='Epochs', total=self.epochs,
                              disable=not self.show_progressbar, unit='epoch', initial=self.start_epoch)
        
        for epoch in epoch_iterator:
            train_loss, validate_loss = self._train()
            self.train_loss = train_loss
            self.validate_loss = validate_loss

            if self.best_loss is None:
                self.best_loss = validate_loss
                if self.save_best:
                    self.best_model = self.model.state_dict()
            elif validate_loss >= self.best_loss:
                self.counter += 1
                if self.counter >= self.patience:
                    break
            else:
                self.best_loss = validate_loss
                if self.save_best:
                    self.best_model = self.model.state_dict()
                self.counter = 0
            
            epoch_iterator.set_postfix_str(f'best loss: {self.best_loss:.7f}')
        
        if self.save_best:
            return self.best_loss, self.best_model
        else:
            return self.best_loss