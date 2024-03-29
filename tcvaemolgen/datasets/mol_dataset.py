import argparse
import collections
import rdkit.Chem as Chem
import sys
import torch.utils.data as data

from tcvaemolgen.utils import path_utils
import pdb

class MolDataset(data.Dataset):
    def __init__(self, raw_data, split_indices, args):
        self.args = args
        self.cache = collections.defaultdict()

        if split_indices is not None:
            data = []
            for split_index in split_indices:
                data.append(raw_data[split_index])
        else:
            data = raw_data
        self.data = data

    def __getitem__(self, index):
        smiles, label = self.data[index]
        try:
            mol = self.cache[smiles]
        except KeyError:
            mol = Chem.MolFromSmiles(smiles)
            self.cache[smiles] = mol
            
        n_atoms = mol.GetNumAtoms()

        path_input = None
        path_mask = None
        if self.args.use_paths:
            shortest_paths = [self.args.p_info[smiles]]
            path_input, path_mask = path_utils.get_path_input(
                [mol], shortest_paths, n_atoms, self.args, output_tensor=True)
            #print(path_input.squeeze(0).shape)
            path_input = path_input.squeeze(0)  # Remove batch dimension
            path_mask = path_mask.squeeze(0)  # Remove batch dimension
        return smiles, label, n_atoms, (path_input, path_mask)

    def __len__(self):
        return len(self.data)


class Args:
    def __init__(self):
        self.max_path_length = 3
        self.p_embed = True
        self.ring_embed = True

def combine_data(data):
    args = Args()
    batch_smiles, batch_labels, batch_n_atoms, batch_path = zip(*data)

    batch_path_inputs, batch_path_masks = zip(*batch_path)
    

    if batch_path_inputs[0] is not None:  # This means paths are used
        max_atoms = max(batch_n_atoms)
        batch_path_inputs, batch_path_mask = path_utils.merge_path_inputs(
            batch_path_inputs, batch_path_masks, max_atoms, args)
    else:
        batch_path_inputs = None
        batch_path_mask = None
    return batch_smiles, batch_labels, (batch_path_inputs, batch_path_mask)

def get_loader(raw_data, split_indices, args, sampler=None, shuffle=False,
               num_workers=5, batch_size=0):    
    mol_dataset = MolDataset(raw_data, split_indices, args)

    if batch_size == 0:
        batch_size = args.batch_size

    data_loader = data.DataLoader(
        mol_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=combine_data,
        num_workers=num_workers)
    return data_loader