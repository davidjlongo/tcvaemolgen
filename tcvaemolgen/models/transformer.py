__author__ = "David Longo (longodj@gmail.com)"

"""
This file defines the core research contribution   
"""
import logging
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from argparse import ArgumentParser

import pytorch_lightning as pl

from tcvaemolgen.datasets.mol_dataset import get_loader
from tcvaemolgen.structures import MolGraph, MolTree
from tcvaemolgen.structures import mol_features
from tcvaemolgen.utils.data import read_smiles_multiclass, read_splits
from tcvaemolgen.utils import path_utils

module_log = logging.getLogger('tcvaemolgen.transformer')

class MoleculeTransformer(pl.LightningModule):

    def __init__(self, hparams):
        log = logging.getLogger('tcvaemolgen.transformer.MoleculeTransformer')
        super(MoleculeTransformer, self).__init__()
        log.debug('Entering MoleculeTransformer')
        self.log = log
        
        n_atom_feats = mol_features.N_ATOM_FEATS
        n_path_features = path_utils.get_num_path_features(hparams)
        n_score_feats = 2 * hparams.d_k + n_path_features
        
        # Model 
        self.hparams = hparams
        """Input Embedding"""
        self.W_atom_i = nn.Linear(in_features=n_atom_feats, 
                                  out_features=hparams.n_heads * hparams.d_k, 
                                  bias=False)
        """Atom Attention Scores"""
        self.W_attn_h = nn.Linear(in_features=n_score_feats, 
                                  out_features=hparams.d_k)
        self.W_attn_o = nn.Linear(in_features=hparams.d_k, out_features=1)
        """Atom Embedding"""
        self.W_message_h = nn.Linear(in_features=n_score_feats, 
                                     out_features=hparams.d_k)
        
        self.W_atom_o = nn.Linear(n_atom_feats + \
                                    hparams.n_heads * \
                                    hparams.d_k, 
                                  hparams.hidden_size)
        
        self.dropout = nn.Dropout(hparams.dropout)
        self.output_size = hparams.hidden_size
        
    
    def _avg_attn(self, attn_probs, n_heads, batch_sz, max_atoms):
        if n_heads > 1:
            attn_probs = attn_probs.view(n_heads, batch_sz, max_atoms, max_atoms)
            attn_probs = torch.mean(attn_probs, dim=0)
        return attn_probs

    def _compute_attn_probs(self, attn_input, attn_mask, layer_idx, eps=1e-20):
        # attn_scores is [batch, atoms, atoms, 1]
        if False:#self.hparams.no_share:
            attn_scores = nn.LeakyReLU(0.2)(
                self.W_attn_h[layer_idx](attn_input))
            attn_scores = self.W_attn_o[layer_idx](attn_scores) * attn_mask
        else:
            attn_scores = nn.LeakyReLU(0.2)(
                self.W_attn_h(attn_input))
            attn_scores = self.W_attn_o(attn_scores) * attn_mask

        # max_scores is [batch, atoms, 1, 1], computed for stable softmax
        max_scores = torch.max(attn_scores, dim=2, keepdim=True)[0]
        # exp_attn is [batch, atoms, atoms, 1]
        exp_attn = torch.exp(attn_scores - max_scores) * attn_mask
        # sum_exp is [batch, atoms, 1, 1], add eps for stability
        sum_exp = torch.sum(exp_attn, dim=2, keepdim=True) + eps

        # attn_probs is [batch, atoms, atoms, 1]
        attn_probs = (exp_attn / sum_exp) * attn_mask
        return attn_probs

    def _compute_nei_score(self, attn_probs, path_mask):
        # Compute the fraction of attn weights in the neighborhood
        nei_probs = attn_probs * path_mask.unsqueeze(3)
        nei_scores = torch.sum(nei_probs, dim=2)
        avg_score = torch.sum(nei_scores) / torch.sum(nei_scores != 0).float()
        return avg_score.item()
    
    def _convert_to_2D(self, input, scope):
        """Convert back to 2D
        Args:
            input: A tensor of shape [batch size, max padding, # features]
            scope: A list of start/length indices for the molecules
        Returns:
            A matrix of size [# atoms, # features]
        """
        input_2D = []

        for idx, mol in enumerate(scope):
            mol_input = input[idx].narrow(0, 0, len(mol['atoms']))
            input_2D.append(mol_input)

        input_2D = torch.cat(input_2D, dim=0)
        return input_2D
    
    def _convert_to_3D(self, input, scope, max_atoms, self_attn=True):
        """Converts the input to a 3D batch matrix
        Args:
            input: A tensor of shape [# atoms, # features]
            scope: A list of start/length indices for the molecules
            max_atoms: The maximum number of atoms for padding purposes
        Returns:
            A matrix of size [batch_size, max atoms, # features]
        """
        n_features = input.size()[1]

        batch_input = []
        batch_mask = []
        for st, le in enumerate(scope):
            length = len(le['atoms'])
            mol_input = input.narrow(0, st, length)
            if self.on_gpu:
                mol_input = mol_input.cuda()
            n_atoms = length
            n_padding = max_atoms - length

            mask = torch.ones([n_atoms])
            if self.on_gpu:
                mask = mask.cuda()

            if n_padding > 0:
                z_padding = torch.zeros([n_padding])
                z_pad_feats = torch.zeros([n_padding, n_features])
                
                if self.on_gpu:
                    z_padding = z_padding.cuda()
                    z_pad_feats = z_pad_feats.cuda()
                    
                mask = torch.cat(
                    [mask.cuda(), z_padding.cuda()])
                mol_input_padded = torch.cat(
                    [mol_input.cuda(), z_pad_feats.cuda()])
                if self.on_gpu:
                    mask, mol_input_padded = mask.cuda(), \
                                             mol_input_padded.cuda()
                batch_input.append(mol_input_padded)
            else:
                batch_input.append(mol_input)

            # TEST
            mask = mask.repeat([max_atoms, 1]) * mask.unsqueeze(1)
            # mask = mask.repeat([max_atoms, 1])

            if not self_attn:
                for i in range(max_atoms):
                    mask[i, i] = 0
            batch_mask.append(mask.cuda())

        batch_input = torch.stack(batch_input, dim=0)
        batch_mask = torch.stack(batch_mask, dim=0).byte()
        return batch_input, batch_mask
    
    def _get_attn_input(self, atom_h, path_input, max_atoms):
        # attn_input is concatentation of atom pair embeddings and path input
        atom_h1 = atom_h.unsqueeze(2).expand(-1, -1, max_atoms, -1)
        self.log.debug(f'Atom_h1: {atom_h1.shape}')
        atom_h2 = atom_h.unsqueeze(1).expand(-1, max_atoms, -1, -1)
        self.log.debug(f'Atom_h2: {atom_h2.shape}')
        atom_pairs_h = torch.cat([atom_h1, atom_h2], dim=3)
        self.log.debug(atom_pairs_h.size())
        self.log.debug(path_input.size())
        attn_input = torch.cat([atom_pairs_h, path_input], dim=3)

        return attn_input

    def forward(self, mol_graph):
        atom_input, scope = mol_graph.get_atom_inputs()
        max_atoms = len(max(scope, key=lambda x: len(x['atoms']))['atoms'])
        
        
        atom_input_3D, atom_mask = self._convert_to_3D(atom_input, 
                                                       scope, 
                                                       max_atoms, 
                                                       True) #Todo: self.hparams.self_attn
        atom_input_3D = atom_input_3D
        attn_mask = atom_mask.float()
        if self.on_gpu:
            atom_input_3D = atom_input_3D.cuda()
            attn_mask = attn_mask.cuda()
        attn_mask = attn_mask.unsqueeze(3)

                
        path_input, path_mask = mol_graph.path_input, mol_graph.path_mask
        if self.on_gpu:
            path_input, path_mask = path_input.cuda(), path_mask.cuda()
        
        batch_sz, _, _ = atom_input_3D.size()
        n_heads, d_k = self.hparams.n_heads, self.hparams.d_k
        
        #if self.hparams.mask_neigh:
        #    attn_mask = path_mask
        #else:
    
        if n_heads > 1:
            attn_mask = attn_mask.repeat(n_heads, 1, 1, 1)
            path_input = path_input.repeat(n_heads, 1, 1, 1)
            path_mask = path_mask.repeat(n_heads, 1, 1)
            
            
        atom_input_h = self.W_atom_i(atom_input_3D).view(batch_sz, max_atoms, 
                                                         n_heads, d_k).float()
        atom_input_h = atom_input_h.permute(2, 0, 1, 3)
        atom_input_h = atom_input_h.contiguous().view(-1, max_atoms, d_k)   
        
        attn_list, nei_scores = [], []
        
        # atom_h should be [batch_size * n_heads, atoms, # features]
        atom_h = atom_input_h
        for layer_idx in range(self.hparams.depth - 1):
            attn_input = self._get_attn_input(atom_h, path_input, max_atoms)

            attn_probs = self._compute_attn_probs(attn_input, attn_mask, layer_idx)
            attn_list.append(self._avg_attn(attn_probs, n_heads, batch_sz, max_atoms))
            nei_scores.append(self._compute_nei_score(attn_probs, path_mask))
            attn_probs = self.dropout(attn_probs)

            if False:#self.hparams.no_share:
                attn_h = self.W_message_h[layer_idx](
                    torch.sum(attn_probs * attn_input, dim=2))
            else:
                attn_h = self.W_message_h(
                    torch.sum(attn_probs * attn_input, dim=2))
            atom_h = nn.ReLU()(attn_h + atom_input_h)

        # Concat heads
        atom_h = atom_h.view(n_heads, batch_sz, max_atoms, -1)
        atom_h = atom_h.permute(1, 2, 0, 3).contiguous().view(batch_sz, max_atoms, -1)

        atom_h = self._convert_to_2D(atom_h, scope)
        self.log.debug("converting to 2D")
        self.log.debug(atom_input.shape)
        self.log.debug(atom_h.shape)
        atom_output = torch.cat([atom_input, atom_h], dim=1)
        self.log.debug(atom_output.shape)
        atom_out = self.W_atom_o(atom_output)
        atom_h = nn.ReLU()(atom_out)
        
        return atom_h, attn_list

    def training_step(self, batch, batch_idx):
        # REQUIRED
        smiles_list, labels_list, (path_input, path_mask) = batch
        n_data = len(smiles_list)
        mol_graph = MolGraph(smiles_list, self.hparams, path_input, path_mask)
        if self.on_gpu:
            mol_graph.cuda()
        self.log.debug(len(labels_list))
        atom_pairs_idx, labels = zip(*labels_list)

        pred_logits = atom_predictor(
            mol_graph, atom_pairs_idx, stats_tracker).squeeze(1)
        labels = [torch.tensor(x) for x in labels] \
            if not self.on_gpu else \
                [torch.tensor(x).cuda() for x in labels]
        labels = torch.cat(labels, dim=0)

        all_pred_logits.append(pred_logits)
        all_labels.append(labels)

        if hparams.n_classes > 1:
            pred_probs = nn.Softmax(dim=1)(pred_logits)
            loss = F.cross_entropy(input=pred_logits, target=labels)
        else:
            pred_probs = nn.Sigmoid()(pred_logits)
            loss = nn.BCELoss()(pred_probs, labels.float())

        stats_tracker.add_stat('loss', loss.item() * n_data, n_data)
        loss = loss / hparams.batch_splits

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

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
        raw_data = read_smiles_multiclass('%s/raw.csv' % self.hparams.data)
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
        raw_data = read_smiles_multiclass('%s/raw.csv' % self.hparams.data)
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
        parser.add_argument('--n_heads', default=8, type=int)
        parser.add_argument('--n_score_feats', default=8, type=int)
        parser.add_argument('--d_k', default=80, type=int)
        parser.add_argument('--depth', default=5, type=int)
        parser.add_argument('--data', default='data/', type=str)
        parser.add_argument('--dropout', default=0.01, type=float)
        parser.add_argument('--use-paths', default=False, type=bool)
        parser.add_argument('--self_attn', default=True, type=bool)
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