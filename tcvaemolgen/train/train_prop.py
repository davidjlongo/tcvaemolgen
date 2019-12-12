"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
import hashlib
import logging
import numpy as np
import os
import sys
import time
import torch
from applicationinsights import channel
from applicationinsights.logging import LoggingHandler

from pytorch_lightning import Trainer
from pytorch_lightning.logging import CometLogger
from argparse import ArgumentParser

from tcvaemolgen.models.prop_predictor import PropPredictor
from tcvaemolgen.utils.data import load_shortest_paths

from tcvaemolgen.utils.data import read_smiles_multiclass

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)
"""
def trace(frame, event, arg):
    with open('stacktrace.log', 'a+') as f:
        f.write("%s, %s:%d\n" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace

sys.settrace(trace)
"""

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
log.setLevel(logging.ERROR)
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
    raw_data = read_smiles_multiclass('%s/raw.csv' % hparams.data)
    n_classes = len(raw_data[0][1])
    model = PropPredictor(hparams, n_classes=n_classes)
    model.version = hparams.dataset
    
    load_shortest_paths(hparams)
    #model.half()
    
    comet_logger = CometLogger(
        api_key=os.environ["COMET_KEY"],
        experiment_name=f'{hparams.dataset}-{str(time.time())}',
        log_graph=True,
        project_name="tcvaemolgen",
        workspace=os.environ["COMET_WKSP"]
    )

    # most basic trainer, uses good defaults
    trainer = Trainer(
        check_val_every_n_epoch=1,
        default_save_path=f'data/05_model_outputs/{hparams.dataset}',
        distributed_backend=hparams.distributed_backend,
        max_nb_epochs=hparams.max_nb_epochs,
        early_stop_callback=None,
        gpus=hparams.gpus,
        gradient_clip_val=10,
        nb_gpu_nodes=hparams.nodes,
        logger=comet_logger,
        log_save_interval=100,
        row_log_interval=10,    
        show_progress_bar=True,
        track_grad_norm=2
    )
    for round_idx in range(hparams.n_rounds):
        model.split_idx = round_idx
        log.info(f'Split {round_idx}')
        trainer.fit(model)
    
    trainer.test()
    return


if __name__ == '__main__':
    # logging shutdown will cause a flush of all un-sent telemetry items
    # alternatively flush manually via handler.flush()
    logging.shutdown()
    
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                        help='supports three options dp, ddp, ddp2')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--n_classes',type=int,default=1)
    parser.add_argument('--n_workers',type=int,default=5)
    parser.add_argument('--save-path', metavar='DIR', default=".", type=str,
                        help='path to save output')
    parser.add_argument('--use-paths', default=True, type=bool)
    parser.add_argument('--self_attn', default=True, type=bool)
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--use-16bit', dest='use-16bit', action='store_true',
                        help='if true uses 16 bit precision')

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = PropPredictor.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    main(hparams)