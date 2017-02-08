
import numpy as np
import theano
import theano.tensor as T
import synkhronos as synk

synk.fork()
x = T.matrix('x')
y = T.matrix('y')
z = T.mean(x.dot(y), axis=0)
f = synk.function([x, y], z, broadcast_inputs=[y])
synk.distribute()

x_dat = np.random.randn(100, 10).astype('float32')
y_dat = np.random.randn(10, 20).astype('float32')
r = f(x_dat, y_dat)
r_th = f.as_theano(x_dat, y_dat)
assert np.allclose(r, r_th)
