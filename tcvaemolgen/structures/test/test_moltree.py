import pytest

from structures.moltree import MolTree

from utils.test_fixtures import single_smiles, smiles_set
        
def test_node_construction(single_smiles):
    mt = MolTree(single_smiles)
    assert mt.num_nodes == 7
    
def test_edge_construction(single_smiles):
    mt = MolTree(single_smiles)
    assert mt.num_edges == (6<<1)
    
def test_tree(single_smiles):
    ref: Dict = {
        0: {'smiles': 'CO',
            'clique': [1]},
        1: {'smiles':'CC', 
            'clique':[0, 6]},
        2: {'smiles':'CO',
            'clique':[6]},
        3: {'smiles':'CO',
            'clique':[6]},
        4: {'smiles':'CO',
            'clique':[6]},
        5: {'smiles':'CO',
            'clique':[6]},
        6: {'smiles': 'C1CCOCC1',
            'clique':[1, 2, 3, 4, 5]}
    }
    mt = MolTree(single_smiles)
    print(mt.nodes_dict)
    
    assert ref == mt.nodes_dict
    