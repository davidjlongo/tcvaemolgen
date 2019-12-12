import hashlib
import logging
import math
import matplotlib
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os
from PIL import Image
import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw
import sklearn.metrics as metrics
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from argparse import ArgumentParser

import pytorch_lightning as pl

from concurrent.futures import ThreadPoolExecutor
import threading

#from models.mol_conv_net import MolConvNet
from tcvaemolgen.datasets.mol_dataset import get_loader
from tcvaemolgen.models.transformer import MoleculeTransformer
from tcvaemolgen.structures import MolGraph, MolTree
from tcvaemolgen.structures import mol_features
from tcvaemolgen.utils.data import read_smiles_from_file, read_smiles_multiclass,\
                                read_splits, write_props

import pdb

module_log = logging.getLogger('tcvaemolgen.atom_predictor')

def drawmols(smiles_list, logger, batch_idx, current_epoch, mode):
    n_data = len(smiles_list)
    mols = []
    for smiles in smiles_list:
        mols.append(Chem.MolFromSmiles(smiles))
    img:Image = Draw.MolsToGridImage(mols,
            molsPerRow=int(math.ceil(math.sqrt(n_data))),subImgSize=(200,200),
            legends=[x for x in smiles_list])
    logger.experiment.log_image(img,
            name=f'(Epoch {current_epoch}, {mode} Batch: {batch_idx}')
  
def drawheatmaps(heatmaps, batch_sz, logger, step_idx, current_epoch, mode, hparams):
    i = 65
        
    for heatmap in heatmaps:
        max_atoms = list(heatmap[0].shape[0:2])[0]
        #figsize = (batch_sz*max_atoms[0]+1,batch_sz*max_atoms[1]+1)
        h_batch_sz = int(math.ceil(math.sqrt(batch_sz)))
        figsize = (h_batch_sz * max_atoms, h_batch_sz * max_atoms)

        if hparams.loss_type in ['mse', 'mae']:
            figsize = (h_batch_sz * max_atoms, h_batch_sz * max_atoms)
        else:
            figsize = (int(math.ceil(math.sqrt(h_batch_sz))) * max_atoms, 
                        int(math.ceil(math.sqrt(h_batch_sz)) * max_atoms))
        
        #print(f'figsize {figsize}')
        #print(f'max_atoms {max_atoms}')
        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(h_batch_sz, h_batch_sz),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )


        for ax, im in zip(grid, heatmap.squeeze().cpu().numpy()):
            ax.imshow(im, cmap='hot', interpolation='nearest')

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        logger.experiment.log_image(data, 
                                    name=f'Heatmap {mode} {current_epoch} {chr(i)}',
                                    step=step_idx)
        
        i+=1

class PropPredictor(pl.LightningModule):
    def __init__(self, hparams, n_classes=1):
        super(PropPredictor, self).__init__()
        log = logging.getLogger('tcvaemolgen.transformer.PropPredictor')
        log.debug('Entering PropPredictor')
        self.log =  log
        self.hparams = hparams
        self.batch_sz = self.hparams.batch_size
        print(f'LOSS_TYPE: {self.hparams.loss_type}')
        hidden_size = hparams.hidden_size
        self.n_classes = n_classes
        model = MoleculeTransformer(hparams)
        self.all_pred_logits, self.all_labels = [], []
        if self.hparams.distributed_backend == 'dp':
            self.tp = ThreadPoolExecutor(max_workers=12)
        
        if self.on_gpu:
            model = model.cuda()
        self.model = model
        if self.on_gpu:
            self.model = self.model.cuda()

        self.W_p_h = nn.Linear(model.output_size, hidden_size)  # Prediction
        self.W_p_o = nn.Linear(hidden_size, n_classes)

    def _aggregate_atom_h(self, atom_h, scope):
        mol_h = []
        for (st, le) in enumerate(scope):
            cur_atom_h = atom_h.narrow(0, st, len(le['atoms']))
            #self.log.debug(f'CUR_ATOM_H: {cur_atom_h.shape}')

            if True:#self.hparams.agg_func == 'sum':
                mol_h.append(cur_atom_h.sum(dim=0))
            #elif self.hparams.agg_func == 'mean':
            #    mol_h.append(cur_atom_h.mean(dim=0))
            else:
                assert(False)
        mol_h = torch.stack(mol_h, dim=0)
        return mol_h
    
    def _compute_acc(self, input_probs, target, n_classes=1):
        if n_classes > 1:
            preds = np.argmax(input_probs, axis=1)
            acc = np.mean(preds == target)
        else:
            preds = (input_probs > 0.5).astype(int)
            acc = np.mean(preds == target)
        return acc

    def _compute_auc(self, input_probs, target):
        auc = metrics.roc_auc_score(y_true=target, y_score=input_probs)
        return auc

    def forward(self, mol_graph, step_idx, mode='None', output_attn=False):
        attn_list = None
        #if self.hparams.model_type == 'transformer':
        atom_h, attn_list = self.model(mol_graph)
        #else:
        #    atom_h = self.model(mol_graph, stats_tracker)
        
        #at = []
        #for in, row in enumerate(attn_list[0].detach()):
        #    at.append(row.squeeze().cpu().numpy())
        
        heatmaps = [x.detach().clone() for x in attn_list]
        if step_idx % 100 == 0:
            if self.hparams.distributed_backend == 'dp':
                _ = self.tp.submit(drawheatmaps(heatmaps,
                                            self.batch_sz,
                                            self.logger,
                                            step_idx,
                                            self.current_epoch,
                                            mode, 
                                            self.hparams))
            else:
                drawheatmaps(heatmaps,
                                            self.batch_sz,
                                            self.logger,
                                            step_idx,
                                            self.current_epoch,
                                            mode,
                                            self.hparams)
        

        scope = mol_graph.mols
        mol_h = self._aggregate_atom_h(atom_h, scope)
        mol_h = nn.ReLU()(self.W_p_h(mol_h))
        mol_o = self.W_p_o(mol_h)
        
        

        if not output_attn:
            return mol_o
        else:
            return mol_o, attn_list
                
    def step(self, batch, batch_idx, mode='None'):
        smiles_list, labels_list, path_tuple = batch
        path_input, path_mask = path_tuple
        #path_input, path_mask = path_input.squeeze(0), path_mask.squeeze(0)
        if self.hparams.use_paths:
            path_input = path_input
            path_mask = path_mask
            if self.on_gpu:
                path_input, path_mask = path_input.cuda(), path_mask.cuda()

        n_data = len(smiles_list)
        if batch_idx % 500 == 0:
            if self.hparams.distributed_backend == 'dp':
                _ = self.tp.submit(drawmols(smiles_list[:], 
                                            self.logger, 
                                            batch_idx, 
                                            self.current_epoch,
                                            mode))
            else:
                drawmols(smiles_list[:], 
                                            self.logger, 
                                            batch_idx, 
                                            self.current_epoch,
                                            mode)
        
        
        mol_graph = MolGraph(smiles_list, 
                             self.hparams, 
                             path_input, 
                             path_mask, 
                             batch_idx)
        
        pred_logits = self(mol_graph, batch_idx, mode).squeeze(1)
        labels = torch.tensor(labels_list, device='cuda').squeeze()
        
        if self.hparams.loss_type == 'ce':  # memory issues
            self.all_pred_logits.append(pred_logits)
            self.all_labels.append(labels)

        if self.hparams.loss_type == 'mse' or self.hparams.loss_type == 'mae':
            loss = nn.MSELoss()(input=pred_logits, target=labels)
        elif self.hparams.loss_type == 'ce':
            pred_probs = nn.Sigmoid()(pred_logits)
            loss = nn.BCELoss()(pred_probs, labels)
        else:
            print("Improper Loss Type")
            assert(False)
            
        #labels_hat = torch.argmax(labels, dim=0)
        #stats_tracker.add_stat('loss', loss.item() * n_data, n_data)

        loss = loss / self.hparams.batch_splits
        
        acc = torch.sum(
            pred_logits == labels
        ).item() / (n_data * 1.0)
        return loss, acc, loss, (smiles_list, labels_list, pred_logits)
    
    def on_epoch_start(self):
        self.all_pred_logits, self.all_labels = [], []
    
    def on_epoch_end(self):
        if self.hparams.loss_type == 'ce':
            self.all_pred_logits = torch.cat(self.all_pred_logits, dim=0)
            self.all_labels = torch.cat(self.all_labels, dim=0)
            pred_probs = nn.Sigmoid()(self.all_pred_logits).detach().cpu().numpy()
            self.all_labels = self.all_labels.detach().cpu().numpy()
            acc = self._compute_acc(pred_probs, self.all_labels)
            auc = self._compute_auc(pred_probs, self.all_labels)
            self.logger.experiment.log_metric('acc', acc,
                                              epoch=self.current_epoch)
            self.logger.experiment.log_metric('auc', auc,
                                              epoch=self.current_epoch)
            logger_logs = {"loss": auc, "acc": acc}
            self.all_pred_logits, self.all_labels = [], []
        
        self.logger.experiment.log_epoch_end(self.current_epoch)
        self.logger.experiment.send_notification(
            f'Epoch {self.current_epoch} ended')
        
        if self.hparams.loss_type == 'ce':
            return {
                "loss":auc, 
                "progress_bar": logger_logs, 
                "log": logger_logs
            }
        else:
            return
    
    def training_step(self, batch, batch_idx):
        loss, _, mae, _ = self.step(batch, batch_idx, 'Train')
        
        self.logger.experiment.log_metric('train_loss', loss.detach().cpu(),
                                          step=batch_idx, 
                                          epoch=self.current_epoch)
        
        self.logger.experiment.log_metric('train_mae', mae.detach().cpu(),
                                          step=batch_idx, 
                                          epoch=self.current_epoch)

        if self.hparams.loss_type in ['mae', 'mse']:   
            logger_logs = {"loss": loss, "train_mae": mae}
        else:
            logger_logs = {"loss": loss}
        return {
            'loss':loss,
            "progress_bar": logger_logs, 
            "log": logger_logs
        }
    
    def validation_step(self, batch, batch_idx):
        loss, acc, mae, (smiles_list, labels_list, pred_logits) = \
            self.step(batch, batch_idx, 'Val')

        write_path = f'data/05_model_output/{self.hparams.data.split("/")[-1]}_{self.hparams.experiment_label}-valid_{self.current_epoch}'
        if write_path is not None:
            write_props(write_path, smiles_list, labels_list,
                                    pred_logits.cpu().numpy())
            
        self.logger.experiment.log_metric('val_loss', loss.detach().cpu(),
                                          step=batch_idx, 
                                          epoch=self.current_epoch)
        
        self.logger.experiment.log_metric('val_acc', acc,
                                          step=batch_idx, 
                                          epoch=self.current_epoch)
        
        self.logger.experiment.log_metric('val_mae', mae.detach().cpu(),
                                          step=batch_idx, 
                                          epoch=self.current_epoch)

        return {
            'val_loss': loss, 
            'val_acc': torch.tensor(acc),
            'val_mae': mae
            #'smiles_list': smiles_list, 
            #'labels_list': labels_list,
            #'pred_logits': pred_logits
        }
    
    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        mae = torch.stack([x["val_mae"] for x in outputs]).mean()
        """(smiles_list, labels_list, pred_logits) = outputs['smiles_list'],\
                                                outputs['labels_list'],
                                                outputs['pred_logits']"""
        
        
        logger_logs = {"val_acc": avg_acc, "val_loss": avg_loss, "val_mae": mae}
        return {'avg_val_loss': avg_loss, "progress_bar": logger_logs, "log": logger_logs}
    
    def test_step(self, batch, batch_idx):
        loss, acc, mae, _ = self.step(batch, batch_idx, 'T  est')
        
        self.logger.experiment.log_metric('test_loss', loss.detach().cpu(),
                                          step=batch_idx, 
                                          epoch=self.current_epoch)
        self.logger.experiment.log_metric('test_acc', acc,
                                          step=batch_idx, 
                                          epoch=self.current_epoch)
        self.logger.experiment.log_metric('test_mae', mae.detach().cpu(),
                                          step=batch_idx, 
                                          epoch=self.current_epoch)

        return {'test_loss':mae, 'test_acc': torch.tensor(acc), 'test_mae':mae}
    
    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        mae = torch.stack([x['test_mae'] for x in outputs]).mean()
        
        logger_logs = {"test_mae": mae}
        
        return {'avg_test_loss': avg_loss, 'test_mae': mae}
    
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), 
                                lr=self.hparams.learning_rate)
        
    @pl.data_loader
    def train_dataloader(self):
        split_idx = self.split_idx
        # REQUIRED
        train_dir = os.path.join(self.hparams.data, 'train')
        if self.hparams.multi:
            raw_data = read_smiles_multiclass('%s/raw.csv' % self.hparams.data)
            n_classes = len(raw_data[0][1])
            print(f'N_Classes: {n_classes}')
        else:
            raw_data = read_smiles_from_file('%s/raw.csv' % self.hparams.data)
            n_classes = 1
        data_splits = read_splits('%s/split_%d.txt' % (self.hparams.data, split_idx))
        train_dataset = get_loader(raw_data, 
                                   data_splits['train'], 
                                   self.hparams, 
                                   shuffle=True)
        
        if self.use_ddp:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = None
        """
        train_loader = DataLoader(dataset=train_dataset, 
                                  batch_size=self.hparams.batch_size,
                                  shuffle=(train_sampler is None),
                                  num_workers=0,
                                  sampler=train_sampler)
        """
        train_loader = get_loader(raw_data, 
                                   data_splits['train'], 
                                   self.hparams, 
                                   shuffle=True,
                                   sampler=train_sampler, 
                                   batch_size=self.hparams.batch_size,
                                   num_workers=self.hparams.n_workers)
        
        return train_loader

    
    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        val_dir = os.path.join(self.hparams.data, 'valid')
        
        split_idx=0
        if self.hparams.multi:
            raw_data = read_smiles_multiclass('%s/raw.csv' % self.hparams.data)
            n_classes = len(raw_data[0][1])
        else:
            raw_data = read_smiles_from_file('%s/raw.csv' % self.hparams.data)
            n_classes = 1
            
        data_splits = read_splits('%s/split_%d.txt' % (self.hparams.data, split_idx))
        
        val_dataset = get_loader(raw_data, 
                                   data_splits['valid'], 
                                   self.hparams, 
                                   shuffle=True,
                                   batch_size=self.hparams.batch_size,
                                   num_workers=self.hparams.n_workers) #TODO
        
        
        if self.use_ddp:
            val_sampler = DistributedSampler(val_dataset)
        else:
            val_sampler = None

        val_loader = get_loader(raw_data, 
                                   data_splits['valid'], 
                                   self.hparams, 
                                   shuffle=False,
                                   sampler=val_sampler)
        
        return val_loader


    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        test_dir = os.path.join(self.hparams.data, 'test')
        
        split_idx=0
        if self.hparams.multi:
            raw_data = read_smiles_multiclass('%s/raw.csv' % self.hparams.data)
            n_classes = len(raw_data[0][1])
        else:
            raw_data = read_smiles_from_file('%s/raw.csv' % self.hparams.data)
            n_classes = 1
        data_splits = read_splits('%s/split_%d.txt' % (self.hparams.data, split_idx))
        
        test_dataset = get_loader(raw_data, 
                                   data_splits['test'], 
                                   self.hparams, 
                                   num_workers=0,
                                   shuffle=False) #TODO"""
        
        if self.use_ddp:
            test_sampler = DistributedSampler(test_dataset)
        else:
            test_sampler = None

        test_loader = get_loader(raw_data, 
                                  data_splits['test'],
                                  self.hparams,
                                  num_workers=0,
                                  shuffle=(test_sampler is None),
                                  sampler=test_sampler)
        
        return test_loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        hash = hashlib.sha1()
        hash.update(str(time.time()).encode('utf-8'))
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.02, type=float)
        parser.add_argument('--batch_size', default=50, type=int)
        parser.add_argument('--hidden_size', default=160, type=int)
        parser.add_argument('--n_heads', default=2, type=int)
        parser.add_argument('--n_score_feats', default=8, type=int)
        parser.add_argument('--d_k', default=80, type=int)
        parser.add_argument('--data', default='data/', type=str)
        parser.add_argument('--dropout', default=0.2, type=float)
        #parser.add_argument('--use-paths', default=False, type=bool)
        parser.add_argument('--depth', default=5, type=int)
        parser.add_argument('--max_path_length', default=3, type=int)
        parser.add_argument('--p_embed', default=True, type=bool)
        parser.add_argument('--ring_embed', default=True, type=bool)
        parser.add_argument('--no_share', default=True, type=bool)
        parser.add_argument('--no_truncate', default=False, type=bool)
        parser.add_argument('--agg_func', default='sum', type=str)
        parser.add_argument('--batch_splits', type=int, default=1,
                        help='Used to aggregate batches')
        parser.add_argument('--multi', type=bool, default=False)
        
        # training specific (for this model)
        parser.add_argument('--epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=42,
                            help='seed for initializing training. ')
        parser.add_argument('--loss_type', default='mae', type=str)
        parser.add_argument('--max_nb_epochs', default=2, type=int)
        parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                            metavar='LR', help='initial learning rate', 
                            dest='learning_rate')
        #parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            #help='momentum')
        #parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            #metavar='W', help='weight decay (default: 1e-4)',
                            #dest='weight_decay')
        parser.add_argument('--pretrained', dest='pretrained', 
                            action='store_true', help='use pre-trained model')
        parser.add_argument('--experiment_label', 
                            default=hash.hexdigest()[:10])
        parser.add_argument('--n_rounds', default=10, type=int)
        
        

        return parser