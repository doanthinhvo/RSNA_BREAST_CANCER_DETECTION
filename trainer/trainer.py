import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker, gc_collect
from torch.cuda.amp import GradScaler, autocast

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metrics_ftns, optimizer, config, device, positive_target_weight,aux_loss_weight,
                 train_dataloader, valid_dataloader=None, lr_scheduler=None, num_train_batchs_per_epoch=None, scaler=None):
        super().__init__(model, criterion,metrics_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.train_dataloader = train_dataloader
        if num_train_batchs_per_epoch is None:
            self.num_train_batchs_per_epoch = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(train_dataloader)
            self.num_train_batchs_per_epoch = num_train_batchs_per_epoch

        self.valid_dataloader = valid_dataloader
        self.do_validation = self.valid_dataloader is not None
        self.lr_scheduler = lr_scheduler

        self.positive_target_weight = positive_target_weight
        self.aux_loss_weight = aux_loss_weight
        self.scaler = GradScaler() 
        
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        # Loop over the batches of the datasiet.
        for batch_idx, (X, cancer, cols_values) in enumerate(self.train_dataloader):
            cols_values, cancer = cols_values.to(self.device), cancer.to(self.device)

            self.optimizer.zero_grad()
            # TODO: add autocast
            # with autocast():
            cancer_pred, cols_values_pred = self.model(X.to(self.device))
            cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(cancer_pred, cancer.to(float).squeeze(), pos_weight=torch.tensor(self.positive_target_weight).to(self.device))

            # print("cols_value_pred : " , cols_values_pred)
            # print("cols_values shape: " , cols_values)
            # print("cols_values_pred[0]: " , cols_values_pred[0])
            # print("cols_values[:, 0]: " , cols_values[:, 0])
            # print("mmoo : " , torch.nn.functional.cross_entropy(cols_values_pred[0], cols_values[:, 0].squeeze()))
            # print("cols_values shape: " , cols_values)
            # print("cols_values_pred[0] shape: " , type(cols_values_pred[0][0]))
            # print("cols_values_pred[0] shape: " , cols_values_pred[0].shape)
            # print("cols_values[:, 0] shape: " , type(cols_values[:, 0]))
            # cols_loss = torch.mean(torch.stack([torch.nn.functional.cross_entropy(cols_values_pred[i], cols_values[:, i].squeeze()) for i in range(cols_values.shape[-1])]))
            cols_loss = 1 
            loss = cancer_loss + cols_loss * self.aux_loss_weight

            if np.isinf(loss.item()) or np.isnan(loss.item()):
                print('Loss is inf or nan')
                del loss, cancer_loss, aux_loss
                gc_collect()
                continue
            

            # TODO: check logic scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.scaler.update()
            lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else 0.0004
            
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(cancer_pred, cancer))

            if batch_idx == self.num_train_batchs_per_epoch:
                break
            print(f'Epoch: {epoch} [{batch_idx}/{self.num_train_batchs_per_epoch}] Loss: {loss.item()} LR: {lr}')
        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        # this log contain the loss and metric of train and validation of this epoch
        print(log)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, (X, cancer, cols_values) in enumerate(self.valid_dataloader):
                cancer, cols_values = cancer.to(self.device), cols_values.to(self.device)

                X = X.to(self.device)

                cancer_pred, cols_values_pred = self.model(X)
                
                cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(cancer_pred, cancer.to(float).squeeze(), pos_weight=torch.tensor(self.positive_target_weight).to(self.device))
                # cols_loss = torch.mean(torch.stack([torch.nn.functional.cross_entropy(cols_values_pred[i], cols_values[:, i]) for i in range(cols_values.shape[-1])]))
                cols_loss = 1
                loss = cancer_loss + cols_loss * self.aux_loss_weight

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(cancer_pred, cancer))

        return self.valid_metrics.result()
