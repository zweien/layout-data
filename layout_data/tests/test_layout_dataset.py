import os
import torch
from layout_data.data.layout import LayoutDataset, LayoutDatasetH5
from layout_data.utils.np_transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
)


def test_layout_dataset(prepare_data_path):
    tmp_path, num, shape = prepare_data_path
    converted_shape = (64, 64)
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
    dataset = LayoutDataset(
        tmp_path, train=True, transform=trms, target_transform=trms
    )
    assert len(dataset) == num

    load, resp = dataset[0]

    assert isinstance(load, torch.Tensor)
    assert load.shape == (1,) + converted_shape
    assert resp.shape == (1,) + converted_shape
    assert abs(load.mean().item() + 0.5) < 0.1  # test mean


def test_layout_datasetH5(prepare_datah5_path):
    path_dir, fn, num, shape = prepare_datah5_path
    converted_shape = (64, 64)

    trms = Compose(
        [
            Resize(size=converted_shape),
            ToTensor(),
            Normalize(torch.tensor([0.5]), torch.tensor([1.0])),
        ]
    )
    dataset = LayoutDatasetH5(
        path_dir,
        train=True,
        transform=trms,
        target_transform=trms,
        train_fn=fn,
    )
    assert len(dataset) == num

    load, resp = dataset[0]

    assert isinstance(load, torch.Tensor)
    assert load.shape == (1,) + converted_shape
    assert resp.shape == (1,) + converted_shape
    assert abs(load.mean().item() + 0.5) < 0.1  # test mean
