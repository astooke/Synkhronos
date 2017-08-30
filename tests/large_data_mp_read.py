
import multiprocessing as mp
import numpy as np
import time
# import synkhronos
# from synkhronos.shmemarray import NpShmemArray

N = 1000
C = 3
H = 224
W = 224
M = 100
K = 100
G = 8


def main():

    # x = NpShmemArray("float32", (N, C, H, W), )
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
    b.wait()
    for i in range(K):
        idxs = rand_idxs[i]
        t_0 = time.time()
        new_mat = x[idxs]
        t_1 = time.time()
        new_mat += 1  # (do something with it to make sure it's real)
        print(rank, "copy time {}: {:.7f}".format(i, t_1 - t_0))
        b.wait()


if __name__ == "__main__":
    main()
