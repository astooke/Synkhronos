import numpy as np
import theano
import theano.tensor as T
import synkhronos as synk

synk.fork()
x = T.matrix('x')
y = T.vector('y')
z = T.mean(x.dot(y), axis=0)
f_th = theano.function(inputs=[x, y], outputs=z)
f = synk.function(inputs=[x], bcast_inputs=[y], outputs=z)
synk.distribute()

x_dat = np.random.randn(100, 10).astype('float32')
y_dat = np.random.randn(10).astype('float32')
x_synk = synk.data(x_dat)
y_synk = synk.data(y_dat)
r_th = f_th(x_dat, y_dat)
r = f(x_synk, y_synk)

assert np.allclose(r, r_th)
print("All assertions passed.")
