"""
Demonstrate optional function keyword argument "batch",
"minibatch" synk Data, function slicing.
"""

import numpy as np
import theano
import theano.tensor as T

import synkhronos as synk
synk.fork()

# Build simple data-parallel computations (parallel across rows of "x")
x = T.matrix('x')
y = T.matrix('y')
w = T.matrix('w')
z = T.sum((x + w).dot(y), axis=0)

f = synk.function(inputs=[x, w], bcast_inputs=[y], outputs=(z, "sum"))
synk.distribute()

x_dat = 0.01 * np.random.randn(1000, 10).astype(theano.config.floatX)
y_dat = np.random.randn(10, 5).astype(theano.config.floatX)
w_dat = 0.01 * np.random.randn(100, 10).astype(theano.config.floatX)

# Build assortment of subsets of the data to compute on.
# (Can be int, slice, or list (e.g. list for random shuffle))
max_idx_0 = 100
slice_1 = slice(100, 200)  # must specify start and stop (for now)
list_2 = np.random.randint(low=0, high=999, size=100)

r_theano_0 = f.as_theano(x_dat[:max_idx_0], w_dat, y_dat)
r_theano_1 = f.as_theano(x_dat[slice_1], w_dat, y_dat)
r_theano_2 = f.as_theano(x_dat[list_2], w_dat, y_dat)

x_dat_synk = synk.data(x_dat)
y_dat_synk = synk.data(y_dat)
w_dat_synk = synk.data(w_dat, minibatch=True)

# kwarg "batch" selects subset of data, which is then scattered among GPUs.
# it applies to all scattering inputs, but not bcast_inputs
# (minibatch data must have length at least as long as the batch, and will be
# selected starting from index 0, which will correspond to the lowest index
# present in "batch", all indexes shifted)
# BEWARE: reduce operation "avg" instead of "sum" might not produce exactly
# the same result, depending on whether the number of data points divides
# evenly (but it will still compute!).

r_0 = f(x_dat_synk, w_dat_synk, y_dat_synk, batch=max_idx_0)
r_1 = f(x_dat_synk, w_dat_synk, y_dat_synk, batch=slice_1)
r_2 = f(x_dat_synk, w_dat_synk, y_dat_synk, batch=list_2)

assert np.allclose(r_theano_0, r_0)
assert np.allclose(r_theano_1, r_1)
assert np.allclose(r_theano_2, r_2)
print("Batch-selective input math tests passed.")

# Can further slice computation within each GPU.
# BEWARE: may also change result for reduce op "avg", if the number of data
# points in a worker does not evenly divide by num_slices (but will still
# compute!).
r_0_slc = f(x_dat_synk, w_dat_synk, y_dat_synk, batch=max_idx_0, num_slices=4)
r_1_slc = f(x_dat_synk, w_dat_synk, y_dat_synk, batch=slice_1, num_slices=4)
r_2_slc = f(x_dat_synk, w_dat_synk, y_dat_synk, batch=list_2, num_slices=4)

assert np.allclose(r_theano_0, r_0_slc)
assert np.allclose(r_theano_1, r_1_slc)
assert np.allclose(r_theano_2, r_2_slc)
print("Batch-selective, sliced computation math tests passed.")
