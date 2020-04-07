import pytest
import os
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
from layout_data.data.loadresponse import LoadResponse, mat_loader
from layout_data.utils.np_transforms import Compose, Resize, ToTensor, Normalize
from sklearn.model_selection import train_test_split


def generate_data(dir_path, num, shape):
    for i in range(num):
        u = np.random.randn(*shape)
        F = np.random.randn(*shape)
        path = os.path.join(dir_path, f'{i}.mat')
        sio.savemat(path, {'u': u, 'F': F})


def test_LoadResponse(tmp_path):
    num = 10
    shape = (200, 200)
    converted_shape = (64, 64)
    generate_data(tmp_path, num, shape)
    assert len(os.listdir(tmp_path)) == num

    trms = Compose([
        Resize(size=converted_shape),
        ToTensor(),
        Normalize(torch.tensor([0.5]), torch.tensor([1.])),
    ])
    dataset = LoadResponse(tmp_path, mat_loader,
                           extensions=('mat'),
                           transform=trms,
                           target_transform=trms)
    assert len(dataset) == num

    data_train, data_val = train_test_split(dataset, train_size=0.8)  # train/val data split

    assert len(data_train) == 8
    assert len(data_val) == 2

    load, resp = data_train[0]

    assert isinstance(load, torch.Tensor)
    assert load.shape == (1,) + converted_shape
    assert resp.shape == (1,) + converted_shape
    assert abs(load.mean().item() + 0.5) < 0.1  # test mean
