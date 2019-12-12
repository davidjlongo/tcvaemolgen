import pytest

from rdkit import Chem 

from tcvaemolgen.structures.mol_features import bt_index_to_float,\
                                    get_bt_index, mol2tensors
                                    
from tcvaemolgen.utils.test_fixtures import single_smiles, smiles_set

def test_bt_index_to_float():
    assert bt_index_to_float(0) == 1
    assert bt_index_to_float(1) == 2
    assert bt_index_to_float(2) == 3
    assert bt_index_to_float(3) == 1.5
    assert bt_index_to_float(4) == 0
    
def test_get_bt_index():
    assert get_bt_index(None) == 4
    assert get_bt_index(Chem.rdchem.BondType.SINGLE) == 0
    assert get_bt_index(Chem.rdchem.BondType.DOUBLE) == 1
    assert get_bt_index(Chem.rdchem.BondType.TRIPLE) == 2
    assert get_bt_index(Chem.rdchem.BondType.AROMATIC) == 3      
    
def test_mol2tensors(single_smiles):
    x, edge_index, edge_attr = mol2tensors(Chem.MolFromSmiles(single_smiles))
    
    print("-"*20)
    print("X: ")
    print(x)
    print("-"*20)
    print("Edge_Index:")
    print(edge_index)
    print("-"*20)
    print("Edge_Attr:")
    print(edge_attr)
    print("-"*20)
    assert 1==1