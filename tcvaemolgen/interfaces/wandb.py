"""Weights and Biases Adapter.
"""

import wandb

from patterns import Adapter

class WandBExperiment(Adapter):
    """WandBExperiment [summary].
    
    Parameters
    ----------
    Adapter : [type]
        [description]
    """
    def __init__(self, project: str):
        super().__init__(wandb)
        self.init(project=project)