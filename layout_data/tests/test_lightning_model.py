import pytest
import math
from argparse import ArgumentParser
from layout_data.models.fpn.model import FPNModel


def test_fpn_lightning():
    parser = ArgumentParser()
    parser = FPNModel.add_model_specific_args(parser)
    hparams = parser.parse_args()
    hparams.gpus = 1
    hparams.data_root = 'd:/work/layout-dataset'

    model = FPNModel(hparams)
    model.prepare_data()

    dataloader = model.train_dataloader()

    F, u = next(iter(dataloader))
    u_pred = model(F)
    assert u_pred.shape == (hparams.batch_size, 1, 200, 200)

    assert not math.isnan(u_pred.sum().item())
