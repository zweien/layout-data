# -*- encoding: utf-8 -*-
'''
Desc      :   Load Response Dataset.
'''
# File    :   loadresponse.py
# Time    :   2020/04/06 17:24:13
# Author  :   Zweien
# Contact :   278954153@qq.com


import os
import scipy.io as sio
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset


class LoadResponse(VisionDataset):
    """Some Information about LoadResponse dataset"""

    def __init__(self, root,
                 loader, load_name='F', resp_name='u',
                 extensions=None,
                 transform=None, target_transform=None, is_valid_file=None):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.root = root
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


def make_dataset(dir, extensions=None, is_valid_file=None):
    """make_dataset() from torchvision.
    """
    files = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_allowed_extension(x, extensions)

    assert os.path.isdir(dir)
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
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
