
import theano
import theano.tensor as T
import numpy as np
import synkhronos as synk

n_gpu = synk.fork()

# x = T.matrix('x')
x_dat = np.random.randn(100, 10).astype(theano.config.floatX)
y_dat = np.random.randn(10, 5).astype(theano.config.floatX)
x = theano.shared(x_dat, 'x_gpu')
y = theano.shared(y_dat, 'y_gpu')
z = T.mean(x.dot(y), axis=0)

f = synk.function(inputs=[], outputs=z, sliceable_shareds=[x])

synk.distribute()

full_x_dat = np.random.randn(n_gpu * 100, 10).astype(theano.config.floatX)

synk.scatter(x, full_x_dat)

r = f()

