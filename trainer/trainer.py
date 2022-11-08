import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
#from data_loader import cutmix
import wandb
from torch.nn import functional as F
import pandas as pd
import os


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        
        wandb.init(project="Mask-Classification", entity="yon-ninii") # wandb initialization
        wandb.config = { # wandb configuration
        "learning_rate": config['optimizer']['args']['lr'],
        "epochs": config['trainer']['epochs'],
        "batch_size": config['data_loader']['args']['batch_size']
        }
        wandb.run.name = config['trainer']['run_name']
        wandb.run.save()
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('Train_loss', *['Train_' + m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('Val_loss', *['Val_' + m.__name__ for m in self.metric_ftns], writer=self.writer)
        #wandb.watch(self.model, self.criterion, log='all', log_freq=1)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        info = pd.read_csv('/opt/ml/code/Lv1/ENet_Implement/testing.csv')   
        t, i = [], []
        for batch_idx, (data, targets) in enumerate(self.data_loader):
            #target1, target2, lam = targets
            # print(lam)
            # print(target1 * lam)
            # print(target2 * lam)
            t.extend(targets.detach().cpu().numpy())
            imgs, image_labels = data.to(self.device).float(), targets.to(self.device).long() 

            if self.config['cutmix']['type']:
                # generate mixed sample
                mix_decision = np.random.rand()
                if mix_decision < self.config['cutmix']['prob']:
                    imgs, image_labels = cutmix(imgs, image_labels, self.config['cutmix']['beta'])
                image_preds = self.model(imgs.float())  
                if mix_decision < self.config['cutmix']['prob']:
                    loss = self.criterion(image_preds, image_labels[0]) * image_labels[2] + self.criterion(image_preds, image_labels[1]) * (1. - image_labels[2])
                else:
                    loss = self.criterion(image_preds, image_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                image_preds = self.model(imgs.float())
                pred = torch.argmax(image_preds.detach().cpu(), dim=1).numpy()
                i.extend(pred)
                loss = self.criterion(image_preds, image_labels) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            

            #self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('Train_loss', loss.item())
            if not self.config['cutmix']['type']:
                for met in self.metric_ftns:
                    self.train_metrics.update('Train_' + met.__name__, met(image_preds.data, image_labels, self.device))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        '''
        train_acc = log['accuracy']
        train_loss = log['loss']
        wandb.log({'train_acc':train_acc, 'train_loss':train_loss})
        '''
        # print(i, t)
        info['ans'] = i
        info['label'] = t
        save_path = os.path.join('/opt/ml/code/Lv1/ENet_Implement/outputs', f'testing.csv')
        info.to_csv(save_path, index=False)
        log = self.train_metrics.result() # Dict type
        #wandb.log(log)
        
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        wandb.log(log)
        
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device).float(), target.to(self.device).long() 

                output = self.model(data)
                loss = self.criterion(output, target)
                

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('Val_loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update('Val_' + met.__name__, met(output, target, self.device))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')
            
        val_log = self.valid_metrics.result()
        #wandb.log(val_log)
        '''
        val_acc = val_log['accuracy']
        val_loss = val_log['loss']
        wandb.log({'val_acc':val_acc, 'val_loss':val_loss})
        '''
        return val_log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)
    return new_data, targets
