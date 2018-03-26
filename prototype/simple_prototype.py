import os
import numpy as np
import ray
import z5py


ray.init(num_gpus=1)


@ray.remote(num_gpus=1)
class InferenceWorker(object):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ray.get_gpu_ids()))

    def infer(self, input_):
        return np.ones_like(input_, dtype='float32')


@ray.remote
def preprocess(data):
    return data + 0.1


@ray.remote
def load(in_path, bb):
    shape = tuple(b.stop - b.start for b in bb)
    return np.zeros(shape, dtype='uint8')


@ray.remote
def save(data, out_path, out_key, bb):
    try:
        z5py.File(out_path)[out_key][bb] = data
    except RuntimeError as e:
        print(bb)
        raise e
    return 1


@ray.remote
def infer(i, j, k, chunks, worker, out_path, out_key):
    in_path = ''
    bb = np.s_[i*chunks[0]:(i + 1)*chunks[0],
               j*chunks[1]:(j + 1)*chunks[1],
               k*chunks[2]:(k + 1)*chunks[2]]
    d = preprocess.remote(load.remote(in_path, bb))
    d = worker.infer.remote(d)
    # NOTE can't return a ray.ObjectID - in a real example we
    # probably want to do this differently
    return ray.get(save.remote(d, out_path, out_key, bb))


def run_inference():
    worker = InferenceWorker.remote()
    out_path = './out.n5'
    out_key = 'out'
    ds = z5py.File(out_path)[out_key]
    chunks = ds.chunks
    chunks_per_dim = ds.chunks_per_dimension

    results = [ray.get(infer.remote(i, j, k, chunks, worker,
                                    out_path, out_key))
               for i in range(chunks_per_dim[0])
               for j in range(chunks_per_dim[1])
               for k in range(chunks_per_dim[2])]

    print("Finished", sum(results), "jobs")


if __name__ == '__main__':
    run_inference()
