import pytest
import os
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
from layout_data.data.layout import LayoutDataset
from layout_data.utils.np_transforms import Compose, Resize, ToTensor, Normalize


def generate_data(dir_path, num, shape):
    for subdir in ["train", "test"]:
        subdir_path = os.path.join(dir_path, subdir)
        os.mkdir(subdir_path)
        for i in range(num):
            u = np.random.randn(*shape)
            F = np.random.randn(*shape)

            path = os.path.join(subdir_path, f"{i}.mat")
            sio.savemat(path, {"u": u, "F": F})


def test_layout_dataset(tmp_path):
    num = 3
    shape = (200, 200)
    converted_shape = (64, 64)
    generate_data(tmp_path, num, shape)
    assert len(os.listdir(tmp_path)) == 2
    train_path = os.path.join(tmp_path, "train")
    assert len(os.listdir(train_path)) == num

    trms = Compose(
        [
            Resize(size=converted_shape),
            ToTensor(),
            Normalize(torch.tensor([0.5]), torch.tensor([1.0])),
        ]
    )
    dataset = LayoutDataset(tmp_path, train=True, transform=trms, target_transform=trms)
    assert len(dataset) == num

    load, resp = dataset[0]

    assert isinstance(load, torch.Tensor)
    assert load.shape == (1,) + converted_shape
    assert resp.shape == (1,) + converted_shape
    assert abs(load.mean().item() + 0.5) < 0.1  # test mean
