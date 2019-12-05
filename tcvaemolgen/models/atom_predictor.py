import logging
import os
import torch
import torch.nn as nn
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
from tcvaemolgen.utils.data import read_smiles_ring_data, read_splits

import pdb

module_log = logging.getLogger('tcvaemolgen.atom_predictor')

class AtomPredictor(pl.LightningModule):
    def __init__(self, hparams, n_classes=1):
        super(AtomPredictor, self).__init__()
        log = logging.getLogger('tcvaemolgen.transformer.AtomPredictor')
        log.debug('Entering AtomPredictor')
        self.hparams = hparams
        hidden_size = hparams.hidden_size

        model = MoleculeTransformer(hparams)
        self.model = model

        self.W_p_h = nn.Linear(model.output_size, hidden_size)  # Prediction
        self.W_p_o = nn.Linear(hidden_size, n_classes)

    def forward(self, mol_graph, pair_idx, output_attn=False):
        attn_list = None
        atom_h, attn_list = self.model(mol_graph)

        all_pairs_h = []
        scope = mol_graph.scope
        for mol_idx, (st, le) in enumerate(scope):
            cur_atom_h = atom_h.narrow(0, st, len(le['atoms']))
            cur_pair_idx = torch.tensor(pair_idx[mol_idx    ],
                                        device=self.hparams.device)
            n_pairs = cur_pair_idx.size()[0]
            cur_pair_idx = cur_pair_idx.view(-1)

            selected_atom_h = torch.index_select(
                input=cur_atom_h, dim=0, index=cur_pair_idx,)
            n_feats = selected_atom_h.size()[1]

            selected_atom_h = selected_atom_h.view([n_pairs, 2, n_feats])
            atom_pair_h = torch.mean(selected_atom_h, 1)
            all_pairs_h.append(atom_pair_h)

        all_pairs_h = torch.cat(all_pairs_h, dim=0)
        all_pairs_h = nn.ReLU()(self.W_p_h(all_pairs_h))
        all_pairs_o = self.W_p_o(all_pairs_h)

        return all_pairs_o
    
    def training_step(self, batch, batch_idx):
        smiles_list, labels_list, path_tuple = batch
        path_input, path_mask = path_tuple
        if self.hparams.use_paths:
            path_input = path_input.to(self.hparams.device)
            path_mask = path_mask.to(self.hparams.device)

        n_data = len(smiles_list)
        mol_graph = MolGraph(smiles_list, self.hparams, path_input, path_mask)
        atom_pairs_idx, labels = zip(*labels_list)

        pred_logits = self(
            mol_graph, atom_pairs_idx).squeeze(1)
        labels = [torch.tensor(x, device=args.device) for x in labels]
        labels = torch.cat(labels, dim=0)

        all_pred_logits.append(pred_logits)
        all_labels.append(labels)

        if self.hparams.n_classes > 1:
            pred_probs = nn.Softmax(dim=1)(pred_logits)
            loss = F.cross_entropy(input=pred_logits, target=labels)
        else:
            pred_probs = nn.Sigmoid()(pred_logits)
            loss = nn.BCELoss()(pred_probs, labels.float())

        #stats_tracker.add_stat('loss', loss.item() * n_data, n_data)
        loss = loss / self.hparams.batch_splits

        return loss
    
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
        raw_data = read_smiles_ring_data('%s/raw.csv' % self.hparams.data)
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
        raw_data = read_smiles_ring_data('%s/raw.csv' % self.hparams.data)
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
        parser.add_argument('--hidden_size', default=200, type=int)
        parser.add_argument('--n_heads', default=8, type=int)
        parser.add_argument('--n_score_feats', default=8, type=int)
        parser.add_argument('--d_k', default=80, type=int)
        parser.add_argument('--data', default='data/', type=str)
        parser.add_argument('--dropout', default=0.01, type=float)
        #parser.add_argument('--use-paths', default=False, type=bool)
        parser.add_argument('--depth', default=5, type=int)
        parser.add_argument('--max-path-length', default=3, type=int)
        parser.add_argument('--p_embed', default=True, type=bool)
        parser.add_argument('--ring_embed', default=True, type=bool)
        parser.add_argument('--no_truncate', default=False, type=bool)
        
        # training specific (for this model)
        parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=None,
                            help='seed for initializing training. ')
        parser.add_argument('--max_nb_epochs', default=2, type=int)
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
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