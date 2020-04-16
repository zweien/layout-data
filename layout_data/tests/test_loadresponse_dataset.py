import torch
from layout_data.data.loadresponse import (
    LoadResponse,
    mat_loader,
    LoadResponseH5,
)
from layout_data.utils.np_transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
)
from sklearn.model_selection import train_test_split
from layout_data.utils.convert import mat2h5


def test_LoadResponseH5(prepare_data_path):
    tmp_path, num, shape = prepare_data_path
    converted_shape = (64, 64)
    tmp_path = tmp_path / "train"
    h5_path = tmp_path / "h5.h5"
    mat2h5(tmp_path, h5_path)
    trms = Compose(
        [
            Resize(size=converted_shape),
            ToTensor(),
            Normalize(torch.tensor([0.5]), torch.tensor([1.0])),
        ]
    )
    dsh5 = LoadResponseH5(h5_path, transform=trms, target_transform=trms)
    assert len(dsh5) == num

    load, resp = dsh5[0]

    assert isinstance(load, torch.Tensor)
    assert load.shape == (1,) + converted_shape
    assert resp.shape == (1,) + converted_shape
    assert abs(load.mean().item() + 0.5) < 0.1  # test mean


def test_LoadResponse(prepare_data_path):
    tmp_path, num, shape = prepare_data_path
    tmp_path = tmp_path / "train"
    converted_shape = (64, 64)

    trms = Compose(
        [
            Resize(size=converted_shape),
            ToTensor(),
            Normalize(torch.tensor([0.5]), torch.tensor([1.0])),
        ]
    )
    dataset = LoadResponse(
        tmp_path,
        mat_loader,
        extensions=("mat"),
        transform=trms,
        target_transform=trms,
    )
    assert len(dataset) == num

    train_size = 0.8
    data_train, data_val = train_test_split(
        dataset, train_size=train_size
    )  # train/val data split

    assert len(data_train) == int(train_size * num)
    assert len(data_val) == int(num - train_size * num)

    load, resp = data_train[0]

    assert isinstance(load, torch.Tensor)
    assert load.shape == (1,) + converted_shape
    assert resp.shape == (1,) + converted_shape
    assert abs(load.mean().item() + 0.5) < 0.1  # test mean
