import z5py


def prepare():
    f = z5py.File('./out.n5', use_zarr_format=False)
    shape = (100, 100, 100)
    chunks = (10, 10, 10)
    f.create_dataset('out', dtype='float32', compression='gzip', shape=shape, chunks=chunks)


if __name__ == '__main__':
    prepare()
