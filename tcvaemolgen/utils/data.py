import dgl
from dgl.data.utils import download, extract_archive, get_download_dir
import logging
import rdkit.Chem as Chem
from pathlib import Path, PosixPath
import pickle
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

from structures.moltree import MolTree
from structures.vocab import Vocab
import structures.mol_features as mf
from structures.mol_features import N_ATOM_FEATS as ATOM_FDIM_DEC
from structures.mol_features import N_BOND_FEATS as BOND_FDIM_DEC
#from models.modules.gcn import mol2dgl_enc

_url = 'https://s3-ap-southeast-1.amazonaws.com/dgl-data-cn/dataset/jtnn.zip'

def load_shortest_paths(args):
    if args.model_type in ['graph_attn_net', 'transformer']:
        args.use_paths = True
        sp_file = '%s/shortest_paths.p' % args.data

        shortest_paths = pickle.load(open(sp_file, 'rb'))
        args.p_info = shortest_paths  # p info can also include rank information

        print('Shortest Paths loaded')

def read_splits(split_path):
    splits = {}

    split_file = open(split_path, 'r+')
    for line in split_file.readlines():
        data_type = line.strip().split(',')[0]
        split_indices = line.strip().split(',')[1:]
        split_indices = [int(x) for x in split_indices]
        splits[data_type] = split_indices
    return splits


def read_smiles_from_file(data_path):
    smiles_data = []

    data_file = open(data_path, 'r')
    for line in data_file.readlines():
        smiles, label = line.strip().split(',')
        smiles_data.append((smiles, float(label)))
    data_file.close()
    return smiles_data

def read_smiles_multiclass(data_path):
    smiles_data = []

    data_file = open(data_path, 'r')
    for line in data_file.readlines():
        smiles = line.strip().split(',')[0]
        labels = line.strip().split(',')[1:]
        labels = [float(x) for x in labels]

        smiles_data.append((smiles, labels))
    return smiles_data

def read_smiles_ring_data(data_path):
    smiles_data = []

    data_file = open(data_path, 'r')
    for line in data_file.readlines():
        smiles = line.strip().split(',')[0]
        pair_labels = line.strip().split(',')[1:]

        atom_pairs = []
        labels = []
        for pair_label in pair_labels:
            pair_str, label_str = pair_label.split(':')
            pair = [int(x) for x in pair_str.split('-')]
            label = int(label_str)
            atom_pairs.append(pair)
            labels.append(label)
        smiles_data.append((smiles, (atom_pairs, labels)))
    return smiles_data


def read_smiles_from_dir(data_dir):
    smiles_data = {}
    for type in ['train', 'valid', 'test']:
        data_path = '%s/%s.txt' % (data_dir, type)

        data = read_smiles_from_file(data_path)
        smiles_data[type] = data
    return smiles_data


def load_shortest_paths(args):
    if True:#args.model_type in ['graph_attn_net', 'transformer']:
        args.use_paths = True
        sp_file = '%s/shortest_paths.p' % args.data

        shortest_paths = pickle.load(open(sp_file, 'rb'))
        args.p_info = shortest_paths  # p info can also include rank information

        print('Shortest Paths loaded')


