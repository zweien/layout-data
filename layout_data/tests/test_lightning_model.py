import pytest
import math
import numpy as np
import os
import scipy.io as sio
from pathlib import Path
from argparse import ArgumentParser
from layout_data.models.fpn.model import FPNModel


def test_fpn_lightning(prepare_data_path):
    path = prepare_data_path

    parser = ArgumentParser()
    parser = FPNModel.add_model_specific_args(parser)
    hparams, _ = parser.parse_known_args()
    hparams.gpus = 1
    hparams.data_root = path

    model = FPNModel(hparams)
    model.prepare_data()

    dataloader = model.train_dataloader()

    F, u = next(iter(dataloader))
    assert u.shape == (hparams.batch_size, 1, 200, 200)
    u_pred = model(F)
    assert u_pred.shape == (hparams.batch_size, 1, 200, 200)

    assert not math.isnan(u_pred.sum().item())
