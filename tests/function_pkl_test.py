
import argparse
import numpy as np
import theano
import theano.tensor as T
import theano.gpuarray
import lasagne.layers as L
from demos.resnet.common import build_resnet
from lasagne import updates
import pickle
import time
import multiprocessing as mp
import sys
sys.setrecursionlimit(50000)


def main(n_gpu=2):

    barrier = mp.Barrier(n_gpu)
    procs = [mp.Process(target=worker, args=(rank, barrier))
        for rank in range(1, n_gpu)]
    for p in procs:
        p.start()

    theano.gpuarray.use("cuda0")
    print("Building model")
    resnet = build_resnet()

    print("Building training function")
    x = T.ftensor4('x')
    y = T.imatrix('y')

    prob = L.get_output(resnet['prob'], x, deterministic=False)
    loss = T.nnet.categorical_crossentropy(prob, y.flatten()).mean()
    params = L.get_all_params(resnet.values(), trainable=True)

    sgd_updates = updates.sgd(loss, params, learning_rate=1e-4)

    # make a function to compute and store the raw gradient
    f_train = theano.function(inputs=[x, y],
                              outputs=loss,  # (assumes this is an avg)
                              updates=sgd_updates)

    print("Pickling function")
    with open("test_pkl.pkl", "wb") as f:
        pickle.dump(f_train, f, pickle.HIGHEST_PROTOCOL)

    barrier.wait()
    time.sleep(1)
    barrier.wait()
    test_the_function(f_train, name="original")

    for p in procs:
        p.join()


def worker(rank, barrier):

    theano.gpuarray.use("cuda" + str(rank))

    barrier.wait()
    print("Unpickling function (in worker {})".format(rank))
    with open("test_pkl.pkl", "rb") as f:
        f_unpkl = pickle.load(f)
    # f_unpkl.trust_input = True

    barrier.wait()
    test_the_function(f_unpkl, name="unpickled", rank=rank)


def test_the_function(f, name="original", rank=0):
    print("Making synthetic data")
    x_dat = np.random.randn(32, 3, 224, 224).astype("float32")
    y_dat = np.random.randint(low=0, high=1000, size=(32, 1)).astype("int32")

    print("Running {} function".format(name))
    r = 0
    for _ in range(10):
        r += f(x_dat, y_dat)
    t_0 = time.time()
    for _ in range(100):
        r += f(x_dat, y_dat)
    t_1 = time.time()
    print("rank {}: {} function ran in {:,.3f} s".format(rank, name, t_1 - t_0))


if __name__ == "__main__":
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['n_gpu'] = int(sys.argv[1])
    main(**kwargs)
