import pytest

@pytest.fixture
def single_smiles():
    return 'OCC1OC(O)C(O)C(O)C1O'               # Glucose

@pytest.fixture
def smiles_set():
    smiles = [
        'OCC1OC(O)C(O)C(O)C1O',                 # Glucose
        'Cn1cnc2n(C)c(=O)n(C)c(=O)c12',         # Caffeine
        'CC(CC(=O)[O-])O',                      # 2-Ketobutyrate
        'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C'   # Testosterone
    ]
    yield smiles