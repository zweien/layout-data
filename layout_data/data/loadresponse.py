# -*- encoding: utf-8 -*-
"""
Desc      :   Load Response Dataset.
"""
# File    :   loadresponse.py
# Time    :   2020/04/06 17:24:13
# Author  :   Zweien
# Contact :   278954153@qq.com

import os
import scipy.io as sio
import h5py
from torchvision.datasets import VisionDataset


class LoadResponse(VisionDataset):
    """Some Information about LoadResponse dataset"""

    def __init__(
        self,
        root,
        loader,
        load_name="F",
        resp_name="u",
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
    ):
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.loader = loader
        self.load_name = load_name
        self.resp_name = resp_name
        self.extensions = extensions
        self.sample_files = make_dataset(root, extensions, is_valid_file)

    def __getitem__(self, index):
        path = self.sample_files[index]
        load, resp = self.loader(path, self.load_name, self.resp_name)
        if self.transform is not None:
            load = self.transform(load)
        if self.target_transform is not None:
            resp = self.target_transform(resp)
        return load, resp

    def __len__(self):
        return len(self.sample_files)


class LoadResponseH5(VisionDataset):
    def __init__(
        self,
        root,
        load_name="F",
        resp_name="u",
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            root, transform=transform, target_transform=target_transform,
        )
        self.load_name = load_name
        self.resp_name = resp_name
        self.data_info = self._get_info(root)

    def _get_info(self, path):
        """get h5 info
        """
        data_info = {}
        with h5py.File(path, "r") as file:
            for key, value in file.items():
                _len, *shape = value.shape
                data_info[key] = {"len": _len, "shape": shape}
        return data_info

    def __getitem__(self, index):
        with h5py.File(self.root, "r") as file:
            load = file[self.load_name][index]
            resp = file[self.resp_name][index]
        if self.transform is not None:
            load = self.transform(load)
        if self.target_transform is not None:
            resp = self.target_transform(resp)
        return load, resp

    def __len__(self):
        return self.data_info[self.load_name]['len']


def make_dataset(root_dir, extensions=None, is_valid_file=None):
    """make_dataset() from torchvision.
    """
    files = []
    root_dir = os.path.expanduser(root_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file \
                cannot be None or not None at the same time"
        )
    if extensions is not None:

        def is_valid_file(x):
            return has_allowed_extension(x, extensions)

    assert os.path.isdir(root_dir), root_dir
    for root, _, fns in sorted(os.walk(root_dir, followlinks=True)):
        for fn in sorted(fns):
            path = os.path.join(root, fn)
            if is_valid_file(path):
                files.append(path)
    return files


def has_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def mat_loader(path, load_name, resp_name=None):
    mats = sio.loadmat(path)
    load = mats.get(load_name)
    resp = mats.get(resp_name) if resp_name is not None else None
    return load, resp
