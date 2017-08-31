
import multiprocessing as mp
import numpy as np
import time
# import synkhronos
from synkhronos.shmemarray import NpShmemArray

N = 1000
C = 3
H = 224
W = 224
M = 100
K = 100
G = 8


def main():

    x = np.ctypeslib.as_array(mp.RawArray('f', N * C * H * W)).reshape(N, C, H, W)
    print(x.shape)

    b = mp.Barrier(G)

    workers = [mp.Process(target=worker, args=(x, b, rank)) for rank in range(1, G)]
    for w in workers:
        w.start()

    worker(x, b, 0)

    for w in workers:
        w.join()


def worker(x, b, rank):

    rand_idxs = np.random.randint(low=0, high=N, size=(K, M))
    total_t = 0.
    if rank == 0:
        y = NpShmemArray("float32", (N, C, H, W), "test_tag", create=True)
        b.wait()
    else:
        b.wait()
        y = NpShmemArray("float32", (N, C, H, W), "test_tag", create=False)
    b.wait()
    for i in range(K):
        idxs = rand_idxs[i]
        t_0 = time.time()
        new_mat = y[idxs]
        # new_mat = x[idxs]
        t_1 = time.time()
        new_mat += 1  # (do something with it to make sure it's real)
        elapse = t_1 - t_0
        total_t += elapse
        print(rank, "copy time {}: {:.7f}".format(i, elapse))
        b.wait()
    print(rank, "total time: {:.4f}  (data size: {})".format(total_t, new_mat.size))


if __name__ == "__main__":
    main()
