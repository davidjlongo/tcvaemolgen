""" [summary]
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from typing import Tuple

from utils.chem import get_mol

import structures.mol_features as mf
from structures import mol_features
import structures

__author__ = "David Longo (longodj@gmail.com)"

class GraphConvNet(MessagePassing):
    """GraphConvNet [summary].

    Parameters
    ----------
    nn : [type]
        [description]
    """
    def __init__(self,
                 hparams):#): argparse.Namespace):
        super(GraphConvNet, self).__init__(aggr='add')
        self.hparams = hparams

        """Message Passing Network."""
        self.W_msg_i = nn.Linear(mol_features.N_ATOM_FEATS +
                                 mol_features.N_BOND_FEATS,
                                 hparams.hidden_size,
                                 bias=False)


        
        """Dropout."""
        #self.dropout = nn.Dropout(args.dropout)

    """ [summary]

    Returns
    -------
    [type]
        [description]
    """
    def forward(self, mol_graph: structures.MolTree):
        
        mol_line_graph = mol_graph.line_graph(backtracking=False,
                                              shared=True)
        
        n_edges = mol_graph.number_of_edges()
        n_nodes = mol_graph.number_of_nodes()
        n_samples = mol_graph.batch_size
        
        """Run."""
        mol_graph.apply_edges(
            func=lambda edges: {'src_x': edges.src['x']},
        )

        e_repr = mol_line_graph.ndata
        bond_features = e_repr['x']  # torch.Tensor
        source_features = e_repr['src_x']  # torch.tensor

        features = torch.cat([source_features, bond_features], 1)
        msg_input = self.W_msg_i(features)
        mol_line_graph.ndata.update({
            'msg_input': msg_input,
            'msg': F.relu(msg_input),
            'accum_msg': torch.zeros_like(msg_input),
        })
        mol_graph.ndata.update({
            'm': bond_features.new(n_nodes, self.hidden_size).zero_(),
            'h': bond_features.new(n_nodes, self.hidden_size).zero_(),
        })

        for i in range(self.depth - 1):
            mol_line_graph.update_all(
                mpn_loopy_bp_msg,
                mpn_loopy_bp_reduce,
                self.apply_mod,
            )

        mol_graph.update_all(
            mpn_gather_msg,
            mpn_gather_reduce,
            self.gather_updater,
        )

        g_repr = dgl.mean_nodes(mol_graph, 'h')
        
        self.n_samples_total += n_samples
        self.n_nodes_total += n_nodes
        self.n_edges_total += n_edges
        self.n_passes += 1
        
        return g_repr