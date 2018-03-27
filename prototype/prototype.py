import os
import numpy as np
import ray
import z5py


ray.init(num_gpus=4)


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
def load(in_path, block, block_shape):
    return np.zeros(block_shape, dtype='uint8')


@ray.remote
def save(data, out_path, out_key, block, chunks):
    bb = tuple(slice(b, b + c) for b, c in zip(block, chunks))
    z5py.File(out_path)[out_key][bb] = data
    return 1


# TODO specify num gpus and num cpus here ?
@ray.remote
def infer(block_list, chunks, worker, out_path, out_key):
    in_path = ''
    tasks = []
    for block in block_list:
        d = preprocess.remote(load.remote(in_path, block, chunks))
        d = worker.infer.remote(d)
        tasks.append(save.remote(d, out_path, out_key, block, chunks))
    return sum(ray.get(tasks))


# implement main inference loop:
# - each worker is running a given number of blocks and we check for completion after that
# TODO add proper error handling to this ....
def main():
    n_workers = 4
    worker_pool = [InferenceWorker.remote() for _ in range(n_workers)]

    out_path = './out.n5'
    out_key = 'out'
    ds = z5py.File(out_path)[out_key]
    chunks = ds.chunks
    chunks_per_dim = ds.chunks_per_dimension

    blocks = [(i * chunks[0], j * chunks[1], k * chunks[2])
              for i in range(chunks_per_dim[0])
              for j in range(chunks_per_dim[1])
              for k in range(chunks_per_dim[2])]

    n_blocks = len(blocks)
    chunk_per_worker = 10
    block_chunk = n_workers * chunk_per_worker
    n_chunks = n_blocks // block_chunk + 1 if n_blocks % block_chunk != 0 else n_blocks // block_chunk

    # submit the first batch of blocks
    chunk_begin = 0
    sub_blocks = blocks[chunk_begin*block_chunk:(chunk_begin + 1) * block_chunk]
    remaining_ids = [infer.remote(sub_blocks[i * chunk_per_worker:(i + 1) * chunk_per_worker],
                                  chunks, worker_pool[i], out_path, out_key)
                     for i in range(n_workers)]

    timeout = 20

    finished = []

    for chunk_begin in range(1, n_chunks):
        ready_ids, remaining_ids = ray.wait(remaining_ids, timeout=timeout)

        # TODO error handling
        # TODO need mapping of object ids to workers to do this properly
        # do something if not all workers have succeeded
        if len(ready_ids) != n_workers:
            pass

        finished.extend(ready_ids)

        sub_blocks = blocks[chunk_begin * block_chunk:(chunk_begin + 1) * block_chunk]
        remaining_ids.extend([infer.remote(sub_blocks[i * chunk_per_worker:(i + 1) * chunk_per_worker],
                                           chunks, worker_pool[i], out_path, out_key)
                              for i in range(n_workers)])

    n_done = sum(ray.get(finished))
    print("Finished", n_done, "tasks")


if __name__ == '__main__':
    main()
