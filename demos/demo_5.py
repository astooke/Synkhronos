"""
Demonstrate optional keyword argument "batch_s" and function slicing
"""

import numpy as np
import theano
import theano.tensor as T

import synkhronos as synk
n_gpus = synk.fork()

DAT = 200  # (data length on each GPU)

# Build simple data-parallel computations with shraed variables.
s_x = theano.shared(np.empty([DAT, 10], dtype=theano.config.floatX))
s_y = theano.shared(np.empty([10, 5], dtype=theano.config.floatX))
z = T.sum(s_x.dot(s_y), axis=0)

f = synk.function(inputs=[], outputs=(z, "sum"), sliceable_shareds=[s_x])
synk.distribute()

x_dat = 0.01 * np.random.randn(DAT * n_gpus, 10).astype(theano.config.floatX)
y_dat = np.random.randn(10, 5).astype(theano.config.floatX)

synk.scatter(s_x, x_dat)
synk.broadcast(s_y, y_dat)

# Build an assortment of subsets of the data to compute on.
# (Can either build a single slice or single list, which will be applied within
# each GPU (AFTER the data is scattered), or can build a list of slices or lists,
# one for each GPU.)
slice_1 = slice(100, 200)
list_2 = np.random.randint(low=0, high=DAT, size=100)
slices_3 = [slice(0 + i, 100 + i) for i in range(n_gpus)]
lists_4 = [np.random.randint(low=0, high=DAT, size=100) for _ in range(n_gpus)]

# Build a Theano function (single-GPU) to check the math.
x = T.matrix('x')
y = T.matrix('y')
z = T.sum(x.dot(y), axis=0)
f_theano = theano.function([x, y], z)

# This shows the data subsets that are selected by the above slices and lists.
x_1 = np.concatenate(
    [x_dat[i * DAT + slice_1.start:i * DAT + slice_1.stop] for i in range(n_gpus)]
)
x_2 = np.concatenate([x_dat[i * DAT + list_2] for i in range(n_gpus)])
x_3s = list()
for i, slc in enumerate(slices_3):
    x_3s.append(x_dat[i * DAT + slc.start:i * DAT + slc.stop])
x_3 = np.concatenate(x_3s)
x_4 = np.concatenate([x_dat[i * DAT + lst] for i, lst in enumerate(lists_4)])

# (single-GPU calls)
r_theano_1 = f_theano(x_1, y_dat)
r_theano_2 = f_theano(x_2, y_dat)
r_theano_3 = f_theano(x_3, y_dat)
r_theano_4 = f_theano(x_4, y_dat)

# kwarg "batch_s" applies within each worker, data has already been scattered
# (unlike kwarg "batch" which acts first on the CPU data set and then scatters)
# (multi-GPU calls)
r_1 = f(batch_s=slice_1)
r_2 = f(batch_s=list_2)
r_3 = f(batch_s=slices_3)
r_4 = f(batch_s=lists_4)

assert np.allclose(r_1, r_theano_1)
assert np.allclose(r_2, r_theano_2)
assert np.allclose(r_3, r_theano_3)
assert np.allclose(r_4, r_theano_4)
print("Batch-selection GPU variable math tests passed.")

# Can further have each GPU compute over slices of its shared variable.
# BEWARE: reduce operation "avg" might yield a slightly different result, if
# worker's data does not divide evenly (but it will still compute!)
r_1_slc = f(batch_s=slice_1, num_slices=4)
r_2_slc = f(batch_s=list_2, num_slices=4)
r_3_slc = f(batch_s=slices_3, num_slices=4)
r_4_slc = f(batch_s=lists_4, num_slices=4)

assert np.allclose(r_1_slc, r_theano_1)
assert np.allclose(r_2_slc, r_theano_2)
assert np.allclose(r_3_slc, r_theano_3)
assert np.allclose(r_4_slc, r_theano_4)
print("Batch-selection, sliced computation, GPU variable math tests passed.")
