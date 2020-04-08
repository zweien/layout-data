"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser
import numpy as np
import torch

import pytorch_lightning as pl

from layout_data.models.fpn.model import FPNModel


def main(hparams):
    """
    Main training routine specific for this project
    """
    seed = hparams.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

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
        precision=16 if hparams.use_16bit else 32,
        val_check_interval=hparams.val_check_interval
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

    parser = configargparse.ArgParser(default_config_files=['config.yml'],
                                      description='Hyper-parameters.')
    parser.add_argument('--config', is_config_file=True,
                        default=False, help='config file path')

    # args

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='seed'
    )

    parser.add_argument(
        '--gpus',
        type=int,
        default=0,
        help='how many gpus'
    )

    parser.add_argument(
        '--use_16bit',
        action='store_true',
        help='use 16 bit precision'
    )

    parser.add_argument(
        '--val_check_interval',
        type=float,
        default=1,
        help='how often within one training epoch to check the validation set'
    )

    parser.add_argument('--test_args', action='store_true',
                        default=False, help='print args')

    parser = FPNModel.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # test args in cli
    if hparams.test_args:
        print(hparams)
    else:
        main(hparams)
