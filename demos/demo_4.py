
import numpy as np
import theano
import theano.tensor as T

import synkhronos as synk
synk.fork()

# Build simple data-parallel computations (parallel across rows of "x")
x = T.matrix('x')
y = T.matrix('y')
z = T.mean(x.dot(y), axis=0)

f = synk.function(inputs=[x], bcast_inputs=[y], outputs=z)
synk.distribute()

x_dat = 0.01 * np.random.randn(1000, 10).astype(theano.config.floatX)
y_dat = np.random.randn(10, 5).astype(theano.config.floatX)

max_idx_0 = 100
slice_1 = slice(10, 100)
slice_2 = slice(100, 1000)
list_3 = np.random.randint(low=0, high=999, size=100)

r_theano_0 = f.as_theano(x_dat[:max_idx_0], y_dat)
r_theano_1 = f.as_theano(x_dat[slice_1], y_dat)
r_theano_2 = f.as_theano(x_dat[slice_2], y_dat)
r_theano_3 = f.as_theano(x_dat[list_3], y_dat)

x_dat_synk, y_dat_synk = f.build_inputs(x_dat, y_dat)

r_0 = f(x_dat_synk, y_dat_synk, batch=max_idx_0)
r_1 = f(x_dat_synk, y_dat_synk, batch=slice_1)
r_2 = f(x_dat_synk, y_dat_synk, batch=slice_2)
r_3 = f(x_dat_synk, y_dat_synk, batch=list_3)

assert np.allclose(r_theano_0, r_0)
assert np.allclose(r_theano_1, r_1)
assert np.allclose(r_theano_2, r_2)
assert np.allclose(r_theano_3, r_3)
print("All batch-selective math tests pasts.")
