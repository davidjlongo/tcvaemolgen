"""Molecular Graph.
"""
import itertools
import logging
import numpy as np
import torch as torch
from typing import Dict, List

from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import os,sys,inspect
sys.path.insert(0,'/home/icarus/app/src') 
from utils.chem import get_clique_mol, tree_decomp, get_mol, get_smiles, \
                       set_atommap, enum_assemble_nx, decode_stereo

from .mol_features import get_atom_features, get_bond_features, mol2tensors, \
                            N_ATOM_FEATS, N_BOND_FEATS, MAX_NEIGHBORS
from .vocab import Vocab

module_log = logging.getLogger('tcvaemolgen.molgraph')
module_log.setLevel(logging.ERROR)

""" MolGraph

Inputs
-------
smiles_list: List[str]
    List of SMILES in batch
    
Returns
-------
[type]
    [description]
"""
class MolGraph(Data):
    def __init__(self, 
                 smiles_list: List[str], 
                 hparams, 
                 path_input=None,
                 path_mask=None,
                 on_gpu:bool=True):
        if smiles_list is None:
            return
        
        self.hparams = hparams
        self.on_gpu = on_gpu
        self.smiles_list = smiles_list
        
        self.mols : List[Data] = []
        
        self.path_input = path_input
        self.path_mask = path_mask
        self.scope = []
        self.rd_mols = []
        
        self._parse_molecules(smiles_list)
        
    def _parse_molecules(self, smiles_list):
        """Turn the smiles into atom and bonds through rdkit.
        Every bond is recorded as two directional bonds, and for each atom,
            keep track of all the incoming bonds, since these are necessary for
            aggregating the final atom feature output in the conv net.
        Args:
            smiles_list: A list of input smiles strings. Assumes that the given
                strings are valid.
            max_atoms: If provided, truncate graphs to this size.
        """
        def skip_atom(atom_idx, max):
            return (max != 0) and (atom_idx >= max)
        
        a_offset = 0
        
        graph_atoms, graph_bonds, graph_attr = None, None, None
        
        for smiles in smiles_list:
            rd_mol = Chem.MolFromSmiles(smiles)
            self.rd_mols.append(rd_mol)
            
            atoms, bonds, attr = mol2tensors(rd_mol)
            if self.on_gpu:
                atoms, bonds, attr = \
                    atoms.cuda(), bonds.cuda(), attr.cuda()

            graph_atoms = torch.cat((graph_atoms, atoms), 0) \
                            if graph_atoms is not None else atoms
                            
            graph_bonds = torch.cat((graph_bonds, bonds), 0)\
                            if graph_bonds is not None else bonds
            graph_attr = torch.cat((graph_attr, attr), 0) \
                            if graph_attr is not None else attr
                            
            if self.on_gpu:
                graph_atoms, graph_bonds, graph_attr = \
                    graph_atoms.cuda(), graph_bonds.cuda(), graph_attr.cuda()

            self.mols.append(dict({
                'atoms':atoms, 
                'bonds':bonds, 
                'attr':attr
            }))
                
        
    def get_atom_inputs(self, output_tensors=True):
        fatoms = []
        
        for mol_idx, mol in enumerate(self.mols):
            atoms = self.rd_mols[mol_idx]
            for atom_idx, atom in enumerate(atoms.GetAtoms()):
                atom_features = get_atom_features(atom)
                fatoms.append(atom_features)
        
        fatoms = np.stack(fatoms, axis=0)
        if output_tensors:
            fatoms = torch.tensor(fatoms).float()
            if self.on_gpu:
                fatoms = fatoms.cuda()
        return fatoms, self.mols
    
    def get_graph_inputs(self):
        fatoms = []
        fbonds = [np.zeros(n_atom_feats + n_bond_feats)]
        agraph = []
        bgraph = [np.zeros([1, max_neighbors])]
        b_offset = 1
        
        for mol_idx, mol in enumerate(self.mols):
            atoms, bonds = mol.atoms, mol.bonds
            cur_agraph = np.zeros([len(atoms), max_neighbors])
            cur_bgraph = np.zeros([len(bonds), max_neighbors])

            for atom_idx, atom in enumerate(atoms):
                atom_features = mol_features.get_atom_features(atom)
                fatoms.append(atom_features)
                for nei_idx, bond in enumerate(atom.bonds):
                    cur_agraph[atom.idx, nei_idx] = bond.idx + b_offset
            for bond in bonds:
                out_atom = atoms[bond.out_atom_idx]
                bond_features = np.concatenate([
                    mol_features.get_atom_features(out_atom),
                    mol_features.get_bond_features(bond)], axis=0)
                fbonds.append(bond_features)
                for i, in_bond in enumerate(out_atom.bonds):
                    if bonds[in_bond.idx].out_atom_idx != bond.in_atom_idx:
                        cur_bgraph[bond.idx, i] = in_bond.idx + b_offset

            agraph.append(cur_agraph)
            bgraph.append(cur_bgraph)
            b_offset += len(bonds)

        fatoms = torch.tensor(np.stack(fatoms, axis=0)).float()
        fbonds = torch.tensor(np.stack(fbonds, axis=0)).float()
        agraph = torch.tensor(np.concatenate(agraph, axis=0)).long()
        bgraph = torch.tensor(np.concatenate(bgraph, axis=0)).long()
        
        if self.on_gpu:
            fatoms = fatoms.cuda()
            fbonds = fbonds.cuda()
            agraph = agraph.cuda()
            bgraph = bgraph.cuda()

        graph_inputs = [fatoms, fbonds, agraph, bgraph]
        return (graph_inputs, self.scope)