# encoding: utf-8

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from .fpn_head import FPNDecoder
from .resnet import resnet50
from .model_init import weights_init
from ..scheduler import WarmupMultiStepLR
from pytorch_lightning import LightningModule
from layout_data.data.layout import LayoutDataset
import layout_data.utils.np_transforms as transforms
from sklearn.model_selection import train_test_split


class FPNModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self._build_model()
        self.criterion = nn.L1Loss()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _build_model(self):
        self.backbone = resnet50()
        self.head = FPNDecoder(encoder_channels=[2048, 1024, 512, 256])
        self.backbone.apply(weights_init)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        # x = torch.sigmoid(x)
        return x

    def __dataloader(self, dataset):
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = WarmupMultiStepLR(optimizer, milestones=[],
                                      warmup_iters=math.ceil(len(self.train_dataset) / self.hparams.batch_size))
        return [optimizer], [scheduler]

    def prepare_data(self):
        """Prepare dataset
        """
        size: int = self.hparams.input_size
        transform_layout = transforms.Compose([
            transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
            transforms.Normalize(torch.tensor(
                [self.hparams.mean_layout]), torch.tensor([self.hparams.std_layout])),
        ])
        transform_heat = transforms.Compose([
            transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
            transforms.Normalize(torch.tensor(
                [self.hparams.mean_heat]), torch.tensor([self.hparams.std_heat])),
        ])
        train_dataset = LayoutDataset(
            self.hparams.data_root, train=True, transform=transform_layout, target_transform=transform_heat)
        test_dataset = LayoutDataset(
            self.hparams.data_root, train=False, transform=transform_layout, target_transform=transform_heat)

        # train/val split
        train_size = int(self.hparams.train_size * len(train_dataset))
        lengths = [train_size, len(train_dataset) - train_size]
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, lengths)

        print(f'Prepared dataset, train:{len(train_dataset)}, val:{len(val_dataset)}, test:{len(test_dataset)}')

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset)

    def val_dataloader(self):
        return self.__dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.__dataloader(self.test_dataset)

    def training_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pred = self(layout)
        loss = self.criterion(heat, heat_pred)
        log = {'training_loss': loss}

        if batch_idx == 0:
            grid = torchvision.utils.make_grid(heat_pred[:4, ...], normalize=True)
            self.logger.experiment.add_image('train_pred_heat_field', grid, self.global_step)
            if self.global_step == 0:
                grid = torchvision.utils.make_grid(heat[:4, ...], normalize=True)
                self.logger.experiment.add_image('train_heat_field', grid, self.global_step)

        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pred = self(layout)
        loss = self.criterion(heat, heat_pred)

        # pred heat field
        grid = torchvision.utils.make_grid(heat_pred[:4, ...], normalize=True)
        self.logger.experiment.add_image('val_pred_heat_field', grid, self.global_step)

        # true layoutand heat field
        if self.global_step == 0 and batch_idx == 0:
            grid = torchvision.utils.make_grid(heat[:4, ...], normalize=True)
            self.logger.experiment.add_image('val_heat_field', grid, self.global_step)

            grid = torchvision.utils.make_grid(layout[:4, ...], normalize=True)
            self.logger.experiment.add_image('val_layout_field', grid, self.global_step)
            
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val_loss': val_loss_mean}
        return {'val_loss': val_loss_mean, 'log': log}

    def test_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pred = self(layout)
        loss = self.criterion(heat, heat_pred)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        log = {'test_loss': test_loss_mean}
        return {'test_loss': test_loss_mean, 'log': log}

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no-cover
        """
        Parameters you define here will be available to your model through `self.hparams`.
        """
        parser = parser

        # dataset args
        parser.add_argument('--data_root', type=str, default='d:/work/dataset')
        parser.add_argument('--train_size', default=0.8,
                            type=float, help='train_size in train_test_split')

        # network params
        parser.add_argument('--drop_prob', default=0.2, type=float)
        parser.add_argument('--input_size', default=200, type=int)
        parser.add_argument('--mean_layout', default=0, type=float)
        parser.add_argument('--std_layout', default=1, type=float)
        parser.add_argument('--mean_heat', default=0, type=float)
        parser.add_argument('--std_heat', default=1, type=float)

        # training params (opt)
        parser.add_argument('--max_epochs', default=20, type=int)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--lr', default='0.01', type=float)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--num_workers', default=2, type=int, help='num_workers in DatasetLoader')
        return parser
