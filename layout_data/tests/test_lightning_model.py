import pytest
from argparse import ArgumentParser
import pytorch_lightning as pl
from layout_data.models.fpn.model import FPNModel

def test_fpn_lightning():
    parser = ArgumentParser()
    parser = FPNModel.add_model_specific_args()
    hparams = parser.parse_args()

    model = FPNModel(hparams)
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        precision=16 if hparams.use_16bit else 32,
    )

    # trainer.fit(model)