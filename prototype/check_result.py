import numpy as np
import z5py


def check_result():
    f = z5py.File('./out.n5', use_zarr_format=False)
    d = f['out'][:]
    assert np.allclose(d, 1)
    print("Passed")


if __name__ == '__main__':
    check_result()
