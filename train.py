
"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch

import pytorch_lightning as pl

from layout_data.models.fpn.model import FPNModel


SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = FPNModel(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        precision=16 if hparams.use_16bit else 32,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    # root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(add_help=False)
    
    # dataset args
    parser.add_argument('--data_root', type=str, default='data')

    # gpu args
    parser.add_argument(
        '--gpus',
        type=int,
        default=0,
        help='how many gpus'
    )

    
    parser.add_argument(
        '--distributed_backend',
        type=str,
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )
    parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )

    parser.add_argument('--test', action='store_true', default=False, help='print args')
    parser = FPNModel.add_model_specific_args(parser)

    hparams = parser.parse_args()

    # test args
    if hparams.test:
        print(hparams)
    else:
        main(hparams)