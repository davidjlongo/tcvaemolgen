import copy
from rdkit import Chem
from typing import List,Tuple

def get_slots(smiles: str) -> List[Tuple[str,int,int]]:
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), 
             atom.GetFormalCharge(), 
             atom.GetTotalNumHs()) 
            for atom in mol.GetAtoms()]

class Vocab(object):
    def __init__(self, smiles_list: List[str]):
        self.vocab = smiles_list
        self.vocab.append('C1=CN=NC=C1')
        self.vocab.append('C1=NC=NN=C1')
        self.vocab.append('C1=CN=[NH+]C=C1')
        self.vocab.append('C1=NC=N[NH+]=C1')
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]

    def get_index(self, smiles: str) -> int:
        return self.vmap[smiles]
    
    def get_smiles(self, idx): 
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])
    
    def size(self):
        return len(self.vocab)