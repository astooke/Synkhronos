"""
Demonstrate basic functionality:
building functions,
output reductions,
function call slicing.
"""

import numpy as np
import theano
import theano.tensor as T

import synkhronos as synk
synk.fork()  # processes forked, GPUs initialized

# Build simple data-parallel computations (parallel across rows of "x")
x = T.matrix('x')
y = T.matrix('y')
z_avg = T.mean(x.dot(y), axis=0)
z_sum = T.sum(x.dot(y), axis=0)
z_max = T.max(x.dot(y), axis=0)

# Build Synk function. NOTES:
# 1. bcast_input "y" will have the full value broadcast to all workers
# 2. outputs have different reduce operations (default is "avg")
f = synk.function(inputs=[x],
                  bcast_inputs=[y],
                  outputs=[z_avg, (z_sum, "sum"), (z_max, "max")])
synk.distribute()  # worker GPUs receive all synk functions, prepare to execute

# Generate random data and compute results
x_dat = 0.01 * np.random.randn(1000, 10).astype(theano.config.floatX)
y_dat = np.random.randn(10, 5).astype(theano.config.floatX)

# For comparison, run on only master GPU, as if standard Theano built by:
# f = theano.function(inputs=[x, y], outputs=[z_avg, z_sum, z_max])
r_avg, r_sum, r_max = f.as_theano(x_dat, y_dat)

# Prepare for computation: move data into OS-shared memory (this is one way)
x_dat_synk, y_dat_synk = f.build_inputs(x_dat, y_dat)

# Compute result using multiple GPUs, reduce to master
r_avg_synk, r_sum_synk, r_max_synk = f(x_dat_synk, y_dat_synk)

# Verify results
assert np.allclose(r_avg, r_avg_synk)
assert np.allclose(r_sum, r_sum_synk)
assert np.allclose(r_max, r_max_synk)
print("Basic math test passed.")


# Smaller memory usage: workers compute over slices, then reduce to master
r_avg_slc, r_sum_slc, r_max_slc = f(x_dat_synk, y_dat_synk, num_slices=4)

# Verify results
assert np.allclose(r_avg, r_avg_slc)
assert np.allclose(r_sum, r_sum_slc)
assert np.allclose(r_max, r_max_slc)
print("Sliced math test passed.")

