import torch

from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self,
                 model,
                 optimizer,
                 loss_func,
                 config,
                 show_progressbar,
                 train_dataloader,
                 validate_dataloader) -> None:
        super(Trainer, self).__init__(model, optimizer, loss_func, config, show_progressbar)
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader

        self.device = config['device']
    
    def _train(self):
        train_loss = 0
        train_number = 0
        self.model.train()
        self.model.unfreeze()

        for idx, (X, y) in enumerate(self.train_dataloader):
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            loss = self.loss_func(X, y)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * X.size(0)
            train_number += X.size(0)

        train_loss = train_loss / train_number
        validate_loss = self._validate()
        
        return train_loss, validate_loss
    
    def _validate(self):
        validate_loss = 0
        validate_number = 0
        self.model.eval()
        self.model.freeze()

        with torch.no_grad():
            for idx, (X, y) in enumerate(self.validate_dataloader):
                X = X.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(X)
                loss = self.loss_func.inner_loss(y_hat, y)

                validate_loss += loss.item() * X.size(0)
                validate_number += X.size(0)
            validate_loss = validate_loss / validate_number
        
        return validate_loss