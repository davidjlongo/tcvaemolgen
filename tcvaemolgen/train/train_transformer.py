"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
import hashlib
import logging
import numpy as np
import time
import torch
from applicationinsights import channel
from applicationinsights.logging import LoggingHandler

from pytorch_lightning import Trainer
from argparse import ArgumentParser

from tcvaemolgen.models.transformer import MoleculeTransformer

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


"""Setup Azure Application Insights
 ____ _  _ ____ ____  ___ ____ __ _ ____ 
(  __| \/ |  __|  _ \/ __|  __|  ( (_  _)
 ) _)/ \/ \) _) )   ( (_ \) _)/    / )(  
(____)_)(_(____|__\_)\___(____)_)__)(__) 
 """
# set up channel with context
telemetry_channel = channel.TelemetryChannel()
telemetry_channel.context.application.ver = '0.0.0.0'
#telemetry_channel.context.properties['my_property'] = 'my_value'

# set up logging
az_handler = LoggingHandler('1bd7b388-4afd-4b58-8b2f-060ac172d00d', 
                            telemetry_channel=telemetry_channel)
az_handler.setLevel(logging.DEBUG)
hash = hashlib.sha1()
hash.update(str(time.time()).encode("utf-8", "strict"))
az_handler.setFormatter(
    logging.Formatter(f'{hash.hexdigest()[:16]} ||'
                       '%(name)s - %(levelname)s: %(message)s')
)

f_handler = logging.FileHandler('output.log')
f_handler.setLevel(logging.DEBUG)
f_handler.setFormatter(
    logging.Formatter('%(name)s - %(levelname)s: %(message)s')
)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)#logging.ERROR)
c_handler.setFormatter(
    logging.Formatter('%(name)s - %(levelname)s: %(message)s')
)
log = logging.getLogger('molgen')
log.setLevel(logging.DEBUG)
log.addHandler(az_handler)
log.addHandler(f_handler)
log.addHandler(c_handler)

"""
 ____ ___ _  _  __  ____ _  _  __  __    ___ ____ __ _ 
(_  _) __) )( \/ _\(  __| \/ )/  \(  )  / __|  __|  ( \
  )(( (__\ \/ /    \) _)/ \/ (  O ) (_/( (_ \) _)/    /
 (__)\___)\__/\_/\_(____)_)(_/\__/\____/\___(____)_)__)
"""

def main(hparams):
    # init module
    model = MoleculeTransformer(hparams)
    #model.half()

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
    )
    trainer.fit(model)
    return


if __name__ == '__main__':
    def str2device(string):
        if string == 'cpu':
            return torch.device('cpu')
        if string == 'cuda':
            return torch.device('cuda')

    # log something (this will be sent to the Application Insights service as a trace)
    log.debug('This is a message')
    log.error('This is an error')

    # logging shutdown will cause a flush of all un-sent telemetry items
    # alternatively flush manually via handler.flush()
    logging.shutdown()
    
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                        help='supports three options dp, ddp, ddp2')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--device', type=str2device, default=torch.device('cpu'))
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--save-path', metavar='DIR', default=".", type=str,
                        help='path to save output')
    parser.add_argument('--use-16bit', dest='use-16bit', action='store_true',
                        help='if true uses 16 bit precision')

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = MoleculeTransformer.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    main(hparams)
