import dgl
import torch
import torch.nn as nn
import os

def cuda(tensor: torch.Tensor, args=None):
    if args is not None:
        if args.use_cuda or torch.cuda.is_available():
            return tensor.cuda()
    else:
        if torch.cuda.is_available():
            return tensor.cuda()
        return tensor

def move_dgl_to_cuda(g: dgl.DGLGraph, args=None):
    g.ndata.update({k: cuda(g.ndata[k], args) for k in g.ndata})
    g.edata.update({k: cuda(g.edata[k], args) for k in g.edata})
    
class GRUUpdate(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size

        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

    def update_zm(self, node):
        src_x = node.data['src_x']
        s = node.data['s']
        rm = node.data['accum_rm']
        z = torch.sigmoid(self.W_z(torch.cat([src_x, s], 1)))
        m = torch.tanh(self.W_h(torch.cat([src_x, rm], 1)))
        m = (1 - z) * s + z * m
        return {'m': m, 'z': z}

    def update_r(self, node, zm=None):
        dst_x = node.data['dst_x']
        m = node.data['m'] if zm is None else zm['m']
        r_1 = self.W_r(dst_x)
        r_2 = self.U_r(m)
        r = torch.sigmoid(r_1 + r_2)
        return {'r': r, 'rm': r * m}

    def forward(self, node):
        dic = self.update_zm(node)
        dic.update(self.update_r(node, zm=dic))
        return dic
