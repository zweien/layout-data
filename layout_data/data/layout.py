# -*- encoding: utf-8 -*-
"""
Desc      :   Layout dataset
"""
# File    :   layout.py
# Time    :   2020/04/06 18:02:23
# Author  :   Zweien
# Contact :   278954153@qq.com

import os
from .loadresponse import LoadResponse, mat_loader


class LayoutDataset(LoadResponse):
    """Layout dataset generated by 'layout-generator'.
    """
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None):

        subdir = "train" if train else "test"
        root = os.path.join(root, subdir)
        super().__init__(
            root,
            mat_loader,
            load_name="F",
            resp_name="u",
            extensions="mat",
            transform=transform,
            target_transform=target_transform,
        )
