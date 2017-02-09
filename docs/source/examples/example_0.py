import numpy as np
import theano
import theano.tensor as T
import synkhronos as synk

synk.fork()
x = T.matrix('x')
y = T.matrix('y')
z = T.mean(x.dot(y), axis=0)
f_th = theano.function([x, y], z)
f = synk.function([x, y], z, broadcast_inputs=[y])
synk.distribute()

x_dat = np.random.randn(100, 10).astype('float32')
y_dat = np.random.randn(10, 20).astype('float32')
r_th = f_th(x_dat, y_dat)
r = f(x_dat, y_dat)
r_as_th = f.as_theano(x_dat, y_dat)
assert np.allclose(r, r_th)
assert np.allclose(r_as_th, r_th)
print("All assertions passed.")
