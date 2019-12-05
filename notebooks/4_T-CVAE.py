#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
from dgl import batch, unbatch
import rdkit.Chem as Chem
from rdkit import RDLogger
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
np.random.seed(0)

import os,sys,inspect
sys.path.insert(0,'/home/icarus/app/src') 

from models.jtnn_vae import DGLJTNNVAE
from models.modules import *

#from .nnutils import cuda, move_dgl_to_cuda

#from .jtnn_enc import DGLJTNNEncoder
#from .jtnn_dec import DGLJTNNDecoder
#from .mpn import DGLMPN
#from .mpn import mol2dgl_single as mol2dgl_enc
#from .jtmpn import DGLJTMPN
#from .jtmpn import mol2dgl_single as mol2dgl_dec

import os,sys,inspect
from tqdm import tqdm
sys.path.insert(0,'/home/icarus/T-CVAE-MolGen/src')

from utils.chem import set_atommap, copy_edit_mol, enum_assemble_nx,                             attach_mols_nx, decode_stereo

lg = RDLogger.logger()

lg.setLevel(RDLogger.CRITICAL)


# In[2]:


class TCVAE(nn.Module):
    def __init__(self, vocab, hidden_size, latent_size, depth):
        super(TCVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        
        self.embedding = nn.Embedding(vocab.size(), hidden_size).cuda()
        self.mpn = MPN(hidden_size, depth)
        self.encoder = None  #
        self.decoder = None  #
        
        self.T_mean, self.T_var = nn.Linear(hidden_size, latent_size // 2),                                   nn.Linear(hidden_size, latent_size // 2)
        self.G_mean, self.G_var = nn.Linear(hidden_size, latent_size // 2),                                   nn.Linear(hidden_size, latent_size // 2)
            
        self.n_nodes_total = 0
        self.n_edges_total = 0
        self.n_passes = 0


# ## Posterior
# 
# As in <cite data-cite="7333468/6Y976JUQ"></cite>
# 
# $q(z|x,y) \sim N(\mu,\sigma^2I)$, where
# 
# $\quad\quad h = \text{MultiHead}(c,E_\text{out}^L(x;y),E_\text{out}^L(x;y))$
# 
# $\quad\quad \begin{bmatrix}\mu\\\log(\sigma^2)\end{bmatrix} = hW_q+b_q$

# In[3]:


def sample_posterior(self, prob_decode=False):
    return


# ### Prior
# 
# As in <cite data-cite="7333468/6Y976JUQ"></cite>
# 
# $p_\theta (z|x) \sim N(\mu', \sigma'^2 I)$, where:
# 
# $\quad\quad h' = \text{MultiHead}(c, E_\text{out}^L(x), E_\text{out}^L(x))$
# 
# $\quad\quad \begin{bmatrix}\mu'\\\log(\sigma'^2)\end{bmatrix} = MLP_p(h')$

# In[4]:


def sample_prior(self, prob_decode=False):
    return


# In[5]:


from utils.data import JTNNDataset, JTNNCollator
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from optparse import OptionParser

class ArgsTemp():
    def __init__(self, hidden_size, depth, device):
        self.hidden_size = hidden_size
        self.batch_size = 250
        self.latent_size = 56
        self.depth = depth
        self.device = device
        self.lr = 1e-3
        self.beta = 1.0
        self.use_cuda = torch.cuda.is_available()
        
args = ArgsTemp(200,3, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
print(args.depth)

parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-s", "--save_dir", dest="save_path")
parser.add_option("-m", "--model", dest="model_path", default=None)
parser.add_option("-b", "--batch", dest="batch_size", default=40)
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-z", "--beta", dest="beta", default=1.0)
parser.add_option("-q", "--lr", dest="lr", default=1e-3)
parser.add_option("-e", "--stereo", dest="stereo", default=1)
opts,_ = parser.parse_args()

dataset = JTNNDataset(data='train', vocab='vocab', training=True, intermediates=False)
vocab = dataset.vocab
"""
dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=JTNNCollator(vocab, True, intermediates=False),
        drop_last=True,
        worker_init_fn=None)
        """

model = DGLJTNNVAE(vocab, args.hidden_size, args.latent_size, args.depth, args).cuda()

if opts.model_path is not None:
    model.load_state_dict(torch.load(opts.model_path))
else:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant(param, 0)
        else:
            nn.init.xavier_normal(param)

model.share_memory()
#if torch.cuda.device_count() > 1:
  #print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  #model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)


# In[6]:


MAX_EPOCH = 50
PRINT_ITER = 20

from tqdm import tqdm
from os import access, R_OK
from os.path import isdir
import sys

save_path = '/home/icarus/app/data/05_model_output'
assert isdir(save_path) and access(save_path, R_OK),        "File {} doesn't exist or isn't readable".format(save_path)

def train():
    dataset.training = True
    print("Loading data...")
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=12,
            collate_fn=JTNNCollator(vocab, True),
            drop_last=True,
            worker_init_fn=None)
    dataloader._use_shared_memory = False
    last_loss = sys.maxsize
    print("Beginning Training...")
    for epoch in range(MAX_EPOCH):
        word_acc,topo_acc,assm_acc,steo_acc = 0,0,0,0
        print("Epoch %d: " % epoch)

        for it, batch in tqdm(enumerate(dataloader),total=len(dataloader)):
            model.zero_grad()
            try:
                loss, kl_div, wacc, tacc, sacc, dacc = model(batch, args.beta)
            except:
                print([t.smiles for t in batch['mol_trees']])
                raise
            loss.backward()
            optimizer.step()

            word_acc += wacc
            topo_acc += tacc
            assm_acc += sacc
            steo_acc += dacc

            cur_loss = loss.item()
            
            if (it + 1) % PRINT_ITER == 0:
                word_acc = word_acc / PRINT_ITER * 100
                topo_acc = topo_acc / PRINT_ITER * 100
                assm_acc = assm_acc / PRINT_ITER * 100
                steo_acc = steo_acc / PRINT_ITER * 100

                print("KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f, Loss: %.6f, Delta: %.6f" % (
                    kl_div, word_acc, topo_acc, assm_acc, steo_acc, cur_loss, last_loss - cur_loss))
                word_acc,topo_acc,assm_acc,steo_acc = 0,0,0,0
                sys.stdout.flush()

            if (it + 1) % 1500 == 0: #Fast annealing
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])
                
            if (it + 1) % 100 == 0:
                torch.save(model.state_dict(),
                            save_path + "/model.iter-%d-%d" % (epoch, it + 1))
                      
            #if last_loss - cur_loss < 1e-5:
            #    break
            last_loss = cur_loss

        scheduler.step()
        print("learning rate: %.6f" % scheduler.get_lr()[0])
        torch.save(model.state_dict(), save_path + "/model.iter-" + str(epoch))


# In[ ]:





# In[ ]:





# In[ ]:


import warnings; warnings.simplefilter('ignore')
#train() 


# ## References
# 
# <div class="cite2c-biblio"></div>

# In[ ]:


def test():
    dataset.training = False
    dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=JTNNCollator(vocab, False),
            drop_last=True,
            worker_init_fn=None)#worker_init_fn)

    # Just an example of molecule decoding; in reality you may want to sample
    # tree and molecule vectors.
    for it, batch in enumerate(dataloader):
        #print(batch['mol_trees'])
        gt_smiles = batch['mol_trees'][0].smiles
        #print(gt_smiles)
        model.move_to_cuda(batch)
        _, tree_vec, mol_vec = model.encode(batch)
        tree_vec, mol_vec, _, _ = model.sample(tree_vec, mol_vec)
        smiles = model.decode(tree_vec, mol_vec)
        print(smiles)


# In[ ]:


#torch.cuda.empty_cache()
test()


# In[ ]:





# In[ ]:




