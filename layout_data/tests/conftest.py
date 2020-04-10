import pytest
import numpy as np
import os
import scipy.io as sio


@pytest.fixture(scope="session")
def prepare_data_path(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("data")
    num = 20
    shape = (200, 200)
    for subdir in ["train", "test"]:
        # subdir_path = os.path.join(tmpdir, subdir)
        subdir_path = tmpdir / subdir
        os.mkdir(subdir_path)
        for i in range(num):
            u = np.random.randn(*shape)
            F = np.random.randn(*shape)

            # path = os.path.join(subdir_path, f'{i}.mat')
            path = subdir_path / f"{i}.mat"
            sio.savemat(str(path), {"u": u, "F": F})

    return tmpdir
