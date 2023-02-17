import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, gc_collect


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    # TODO: add device to config
    def __init__(self, model, criterion, optimizer, config, device, positive_target_weight,aux_loss_weight,
                 train_dataloader, valid_dataloader=None, lr_scheduler=None, len_epoch=None, scaler=None):
        super().__init__(model, criterion, optimizer, config)
        
        self.device = device
        self.train_dataloader = train_dataloader
        
        self.valid_dataloader = valid_dataloader
        self.do_validation = self.valid_dataloader is not None
        self.lr_scheduler = lr_scheduler

        self.positive_target_weight = positive_target_weight
        self.aux_loss_weight = aux_loss_weight
        self.scaler = scaler

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()

        # Loop over the batches of the datasiet.
        for batch_idx, (X, cancer, cols_values) in enumerate(self.train_dataloader):
            cols_values, cancer = cols_values.to(self.device), cancer.to(self.device)

            self.optimizer.zero_grad()
            # TODO: add autocast
            # with autocast():
            cancer_pred, cols_values_pred = self.model(X.to(self.device))
            cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(cancer_pred, cancer, pos_weight=torch.tensor(self.positive_target_weight).to(self.device))

            cols_loss = torch.mean(torch.stack([torch.nn.functional.cross_entropy(cols_values_pred[i], cols_values[:, i]) for i in range(cols_values.shape[-1])]))
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

        if self.do_validation:
            val_log = self._valid_epoch(epoch)

        # this log contain the loss and metric of train and validation of this epoch
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, (X, target_cancer, target_cate_aux) in enumerate(self.valid_dataloader):
                target_cancer, target_cate_aux = target_cancer.to(self.device), target_cate_aux.to(self.device)
                X = X.to(self.device)

                pred_cancer, pred_cate_aux = self.model(X)
                loss_cancer = torch.nn.functional.binary_cross_entropy_with_logits(pred_cancer, target_cancer, pos_weight=torch.tensor(self.positive_target_weight).to(self.device))
                loss_cate_aux = torch.mean(torch.stack([torch.nn.functional.cross_entropy(pred_cate_aux[i], target_cate_aux[:, i]) for i in range(target_cate_aux.shape[-1])]))
                loss = loss_cancer + loss_cate_aux * self.aux_loss_weight
        

        # TODO: return loss and metrics in a log 
        return loss

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_dataloader, 'n_samples'):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
