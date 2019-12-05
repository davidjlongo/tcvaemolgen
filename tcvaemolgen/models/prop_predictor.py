import logging
import os
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from argparse import ArgumentParser

import pytorch_lightning as pl

#from models.mol_conv_net import MolConvNet
from tcvaemolgen.datasets.mol_dataset import get_loader
from tcvaemolgen.models.transformer import MoleculeTransformer
from tcvaemolgen.structures import MolGraph, MolTree
from tcvaemolgen.structures import mol_features
from tcvaemolgen.utils.data import read_smiles_from_file, read_splits

import pdb

module_log = logging.getLogger('tcvaemolgen.atom_predictor')

class PropPredictor(pl.LightningModule):
    def __init__(self, hparams, n_classes=1):
        super(PropPredictor, self).__init__()
        log = logging.getLogger('tcvaemolgen.transformer.PropPredictor')
        log.debug('Entering PropPredictor')
        self.log =  log
        self.hparams = hparams
        hidden_size = hparams.hidden_size

        model = MoleculeTransformer(hparams)
        self.model = model

        self.W_p_h = nn.Linear(model.output_size, hidden_size)  # Prediction
        self.W_p_o = nn.Linear(hidden_size, n_classes)

    def aggregate_atom_h(self, atom_h, scope):
        mol_h = []
        for (st, le) in enumerate(scope):
            cur_atom_h = atom_h.narrow(0, st, len(le['atoms']))
            self.log.debug(f'CUR_ATOM_H: {cur_atom_h.shape}')

            if True:#self.hparams.agg_func == 'sum':
                mol_h.append(cur_atom_h.sum(dim=0))
            elif self.hparams.agg_func == 'mean':
                mol_h.append(cur_atom_h.mean(dim=0))
            else:
                assert(False)
        mol_h = torch.stack(mol_h, dim=0)
        return mol_h

    def forward(self, mol_graph, stats_tracker, output_attn=False):
        attn_list = None
        #if self.hparams.model_type == 'transformer':
        atom_h, attn_list = self.model(mol_graph)
        #else:
        #    atom_h = self.model(mol_graph, stats_tracker)

        scope = mol_graph.mols
        mol_h = self.aggregate_atom_h(atom_h, scope)
        mol_h = nn.ReLU()(self.W_p_h(mol_h))
        mol_o = self.W_p_o(mol_h)

        if not output_attn:
            return mol_o
        else:
            return mol_o, attn_list
    
    def training_step(self, batch, batch_idx):
        smiles_list, labels_list, path_tuple = batch
        path_input, path_mask = path_tuple
        if self.hparams.use_paths:
            path_input = path_input.to(self.hparams.device)
            path_mask = path_mask.to(self.hparams.device)

        n_data = len(smiles_list)
        mol_graph = MolGraph(smiles_list, self.hparams, path_input, path_mask)
        pred_logits = self(mol_graph, stats_tracker=None).squeeze(1)
        labels = torch.tensor(labels_list, device=self.hparams.device)

        if False:#self.hparams.loss_type == 'ce':  # memory issues
            all_pred_logits.append(pred_logits)
            all_labels.append(labels)

        if True:#self.hparams.loss_type == 'mse' or self.hparams.loss_type == 'mae':
            loss = nn.MSELoss()(input=pred_logits, target=labels)
        elif self.hparams.loss_type == 'ce':
            pred_probs = nn.Sigmoid()(pred_logits)
            loss = nn.BCELoss()(pred_probs, labels)
        else:
            assert(False)
        #stats_tracker.add_stat('loss', loss.item() * n_data, n_data)
        loss = loss / self.hparams.batch_splits

        return OrderedDict({'loss':loss})
    
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), 
                                lr=self.hparams.learning_rate)
        
    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        train_dir = os.path.join(self.hparams.data, 'train')
        split_idx=0
        raw_data = read_smiles_from_file('%s/raw.csv' % self.hparams.data)
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
                                   shuffle=True)
        
        return train_loader

    """
    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, 
                                transform=transforms.ToTensor()), 
                                batch_size=self.hparams.batch_size)
"""

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        test_dir = os.path.join(self.hparams.data, 'test')
        
        split_idx=0
        raw_data = read_smiles_from_file('%s/raw.csv' % self.hparams.data)
        data_splits = read_splits('%s/split_%d.txt' % (self.hparams.data, split_idx))
        
        test_dataset = get_loader(raw_data, 
                                   data_splits['test'], 
                                   self.hparams, 
                                   shuffle=True) #TODO
        
        if self.use_ddp:
            test_sampler = DistributedSampler(test_dataset)
        else:
            test_sampler = None

        test_loader = DataLoader(dataset=test_dataset, 
                                  batch_size=self.hparams.batch_size,
                                  shuffle=(test_sampler is None),
                                  num_workers=0,
                                  sampler=test_sampler)
        
        return test_loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
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
        parser.add_argument('--no_truncate', default=False, type=bool)
        parser.add_argument('--agg_func', default='sum', type=str)
        parser.add_argument('-batch_splits', type=int, default=1,
                        help='Used to aggregate batches')
        
        # training specific (for this model)
        parser.add_argument('--epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=42,
                            help='seed for initializing training. ')
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
        
        

        return parser