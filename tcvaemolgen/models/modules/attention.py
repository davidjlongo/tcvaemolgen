""" [summary]
"""
import argparse
import numpy as np
import torch as torch
import torch.nn as nn
from typing import Tuple

__author__ = "David Longo (longodj@gmail.com)"


class Attention(nn.Module):
    """Attention [summary].

    Parameters
    ----------
    nn : [type]
        [description]
    """
    def __init__(self,
                 args: argparse.Namespace, 
                 temperature: np.float64, 
                 attn_dropout: np.float64 = 0.1):
        super(Attention, self).__init__()

        self.args = args
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.temperature = temperature

        """Attention Layers."""
        self.W_attn_h = nn.Linear(args.hidden_size, args.hidden_size)
        self.W_attn_o = nn.Linear(args.hidden_size, 1)

    """ [summary]

    Returns
    -------
    [type]
        [description]
    """
    def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        #attn = torch.bmm(q, k.transpose(1, m))
        #output = torch.bmm(attn, v)
        return None, None #output, attn