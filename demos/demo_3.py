"""
Demonstrate interactions with Theano shared variables (GPU memory)
"""

import numpy as np
import theano
import theano.tensor as T
import synkhronos as synk
n_gpus = synk.fork()

# Make data-parallel computation with Theano shared variable (exists on GPU).
dtype = theano.config.floatX
s_x = theano.shared(np.ones([100, 4], dtype=dtype), name='s_x')
s_y = theano.shared(np.zeros([4, 5], dtype=dtype), name='s_y')
s_unused = theano.shared(np.zeros([5, 5], dtype=dtype))  # (see note at bottom)
z = T.mean(s_x.dot(s_y), axis=0)

f = synk.function(inputs=[], sliceable_shareds=[s_x], outputs=z)
synk.distribute()  # (shared variable data sent to workers with function)

# Inspect values of Theano shared variables--separate copy on each GPU.
print("\nLengths of s_x on each GPU: ", synk.get_lengths(s_x))
print("Shapes of s_x on each GPU: ", synk.get_shapes(s_x))

x_dat = np.random.randn(8 * n_gpus, 4).astype(dtype)
y_dat = np.random.randn(4, 5).astype(dtype)

# Manipulate values of Theano shared variables across all GPUs.
synk.scatter(s_x, x_dat)
synk.broadcast(s_y, y_dat)  # (without data arg, operates on existing var data)

print("\nData scattered to s_x and broadcast to s_y...")
print("\nShapes of s_x on each GPU: ", synk.get_shapes(s_x))
gathered_x = synk.gather(s_x, nd_up=0)
assert np.allclose(x_dat, gathered_x)
print("Scatter-gather equivalency test passed.")
print("\nValue of s_x on rank 1:\n", synk.get_value(1, s_x))
ys = synk.gather(s_y)
print("\nValues of s_y on all ranks:")
for i in range(n_gpus):
    print("\n", ys[i])

# Theano for comparison (must compute on GPU for values match).
x_mat = T.matrix('x')
y_mat = T.matrix('y')
z = T.mean(x_mat.dot(y_mat), axis=0)
f_theano = theano.function([x_mat, y_mat], z)
r_theano = f_theano(x_dat, y_dat)

r = f()
assert np.allclose(r, r_theano)
print("\nShared variable math test passed.")

r_slc = f(num_slices=2)  # slicing here works on the GPU variable s_x
assert np.allclose(r_slc, r_theano)
print("Shared variable sliced math test passed.")

# Further manipulations.
# Average values of shared variable on all GPUs:
x_values = synk.gather(s_x, nd_up=1)  # (x_values is on the GPU)
x_values = np.asarray(x_values)  # (now it's on CPU)
synk.all_reduce(s_x, op="avg")  # (default op is "avg")
x_avg = x_values.mean(axis=0)
new_x_values = np.array(synk.gather(s_x, nd_up=1))
for i in range(n_gpus):
    assert np.allclose(x_avg, new_x_values[i])
print("\nValue on rank 1 after all_reduce:\n", synk.get_value(1, s_x))
print("All_reduce avg test passed.")

# Reset one of the GPUs to previous value:
# ipdb.set_trace()
synk.set_value(rank=1, shared_vars=s_x, values=x_values[1])
print("\nReset the value on rank 1:\n", synk.get_value(1, s_x))

# Make all the GPUs have all the data (can't change ndim of variable)
synk.all_gather(s_x)
print("\nShapes of s_x on GPUs after all_gather: ", synk.get_shapes(s_x))

# NOTE: can't call collectives or inspections on "s_unused", because it's not
# used in any function, so the workers do not have it.
