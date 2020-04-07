# encoding: utf-8

import math
from collections import OrderedDict
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from .fpn_head import FPNDecoder
from .resnet import resnet50
from ..scheduler import WarmupMultiStepLR
from ..focalloss import FocalLoss
from pytorch_lightning import LightningModule
import logging
from layout_data.data.layout import LayoutDataset
import layout_data.utils.np_transforms as transforms
from sklearn.model_selection import train_test_split


class FPNModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self._build_model()
        self.criterion = FocalLoss()

    def _build_model(self):
        self.backbone = resnet50()
        self.head = FPNDecoder(encoder_channels=[2048, 1024, 512, 256])

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        # x = torch.sigmoid(x)
        return x

    def __dataloader(self, dataset):
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=0
        )
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = WarmupMultiStepLR(optimizer, milestones=[],
                                      warmup_iters=math.ceil(len(self.train_dataset)/self.hparams.batch_size))
        return [optimizer], [scheduler]

    def prepare_data(self):
        """Prepare dataset
        """
        size: int = self.hparams.input_size
        transform = transforms.Compose([
            transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
            transforms.Normalize(torch.tensor(
                [self.hparams.mean]), torch.tensor([self.hparams.std])),
        ])
        train_dataset = LayoutDataset(
            self.hparams.data_root, train=True, transform=transform, target_transform=transform)
        test_dataset = LayoutDataset(
            self.hparams.data_root, train=True, transform=transform, target_transform=transform)

        # train/val split
        train_dataset, val_dataset = train_test_split(
            train_dataset, train_size=self.hparams.train_size)

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        logging.info('Training data loader called.')
        return self.__dataloader(self.train_dataset)

    def val_dataloader(self):
        logging.info('Validation data loader called.')
        return self.__dataloader(self.val_dataset)

    def test_dataloader(self):
        logging.info('Test data loader called.')
        return self.__dataloader(self.test_dataset)


    def training_step(self, batch, batch_idx):
        F, u = batch
        u_pred = self(F)
        loss = self.criterion(u, u_pred)
        log = {'training_loss': loss}
        return {'loss': loss, 'log': log}



    def validation_step(self, *args, **kwargs):
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Parameters you define here will be available to your model through `self.hparams`.
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)
        
        # dataset args
        parser.add_argument('--data_root', type=str, default='data')
        
        # network params
        parser.add_argument('--drop_prob', default=0.2, type=float)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--input_size', default=200, type=int)
        parser.add_argument('--mean', default=0, type=float)
        parser.add_argument('--std', default=1, type=float)

        # training params (opt)
        parser.add_argument('--max_epochs', default=20, type=int)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--lr', default='0.01', type=float)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--train_size', default=0.8,
                            type=float, help='train_size in train_test_split')
        return parser
